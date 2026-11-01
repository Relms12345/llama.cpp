#include "llama-moe-stream.h"
#include "llama-impl.h"

#include <stdexcept>

#if (defined(__unix__) || defined(__APPLE__)) && defined(GGML_USE_CPU)
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

struct llama_moe_stream::impl {
    struct file {
        int fd;
        explicit file(int source) : fd(dup(source)) {
            if (fd < 0) throw std::runtime_error("expert cache: dup failed: " + std::string(strerror(errno)));
        }
        ~file() { close(fd); }
    };
    struct tensor;
    struct entry;
    using lru_list = std::list<entry>;
    struct entry {
        tensor * owner;
        int32_t expert;
        uint32_t pins = 0;
        void * data;
        lru_list::iterator position;
        entry(tensor * t, int32_t e, void * d) : owner(t), expert(e), data(d) {}
        ~entry() { std::free(data); }
        entry(const entry &) = delete;
        entry & operator=(const entry &) = delete;
    };
    struct usage { uint64_t hits = 0; uint64_t last = 0; };
    struct tensor {
        impl * cache;
        ggml_tensor * weight;
        file * backing;
        uint64_t offset;
        size_t slice;     // bytes of one expert in the file
        size_t pack_work; // in-place repack scratch, 0 = canonical layout
        size_t allocated; // slice + pack_work, 64-byte rounded
        std::vector<entry *> slots;
        std::vector<usage> history;
        ggml_cpu_moe_stream provider;
    };

    mutable std::mutex mutex;
    std::condition_variable available;
    int waiters = 0;
    std::map<std::pair<uint64_t, uint64_t>, std::unique_ptr<file>> files;
    std::map<const ggml_tensor *, std::unique_ptr<tensor>> tensors;
    lru_list lru; // front = most recently used
    llama_moe_cache_stats counters = {};
    uint64_t backing_size = 0;
    std::string profile_path;
    uint64_t identity = 0;
    uint64_t clock = 0;
    bool history_dirty = false;

    explicit impl(uint64_t budget) {
        if (!budget) throw std::runtime_error("expert cache: budget must be positive");
        counters.capacity_bytes = budget;
    }

    ~impl() {
        save_profile();
        for (const auto & t : tensors) ggml_cpu_register_moe_stream(t.first, nullptr);
    }

    // ggml-cpu callbacks: called by one worker per MUL_MAT_ID node, never throw across the C ABI

    static int acquire_cb(void * context, const int32_t * experts, int n, const void ** data) noexcept {
        auto & t = *static_cast<tensor *>(context);
        return t.cache->acquire(t, experts, n, data, true);
    }

    static void release_cb(void * context, const int32_t * experts, int n) noexcept {
        auto & t = *static_cast<tensor *>(context);
        t.cache->release(t, experts, n);
    }

    static void prefetch_cb(void * context, int64_t expert) noexcept {
        auto & t = *static_cast<tensor *>(context);
#if defined(POSIX_FADV_WILLNEED)
        {
            std::lock_guard<std::mutex> lock(t.cache->mutex);
            if (expert < 0 || (size_t) expert >= t.slots.size() || t.slots[expert]) return;
        }
        (void) posix_fadvise(t.backing->fd, (off_t) (t.offset + expert * t.slice), (off_t) t.slice, POSIX_FADV_WILLNEED);
#else
        GGML_UNUSED(t);
        GGML_UNUSED(expert);
#endif
    }

    // all of the following expect the mutex to be held

    void touch(tensor & t, int32_t expert) {
        auto & u = t.history[expert];
        if (u.hits < UINT64_MAX) ++u.hits;
        if (clock < UINT64_MAX) ++clock;
        u.last = clock;
        history_dirty = true;
    }

    // evict unpinned least recently used entries until `bytes` fit
    bool make_room(size_t bytes) {
        auto it = lru.end();
        while (bytes > counters.capacity_bytes - counters.used_bytes) {
            while (it != lru.begin() && std::prev(it)->pins) --it;
            if (it == lru.begin()) return false;
            --it;
            tensor * owner = it->owner;
            owner->slots[it->expert] = nullptr;
            counters.used_bytes -= owner->allocated;
            if (owner->pack_work) counters.repacked_bytes -= owner->allocated;
            it = lru.erase(it);
        }
        return true;
    }

    entry * load(tensor & t, int32_t expert, bool count) {
        lru.emplace_front(&t, expert, nullptr);
        entry & e = lru.front();
        e.position = lru.begin();
        if (posix_memalign(&e.data, 64, t.allocated)) {
            e.data = nullptr;
            lru.pop_front();
            throw std::bad_alloc();
        }
        void * data = e.data;
        const uint64_t base = t.offset + (uint64_t) expert * t.slice;
        const auto start = std::chrono::steady_clock::now();
        size_t done = 0;
        while (done < t.slice) {
            const ssize_t n = pread(t.backing->fd, (char *) data + done,
                    std::min(t.slice - done, (size_t) std::numeric_limits<ssize_t>::max()), (off_t) (base + done));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                const std::string reason = n == 0 ? "unexpected end of file" : strerror(errno);
                lru.pop_front();
                throw std::runtime_error("read failed at file offset " + std::to_string(base + done) + ": " + reason);
            }
            done += n;
        }
#if defined(POSIX_FADV_DONTNEED)
        // the copy is ours now: do not also keep the file pages, that would double the memory use
        (void) posix_fadvise(t.backing->fd, (off_t) base, (off_t) t.slice, POSIX_FADV_DONTNEED);
#endif
        counters.bytes_read += t.slice;
        counters.read_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start).count();
        if (t.pack_work && !ggml_cpu_moe_repack(t.weight, data, (char *) data + t.slice)) {
            lru.pop_front();
            throw std::runtime_error("cannot repack expert");
        }
        if (count) ++counters.misses;
        t.slots[expert] = &e;
        counters.used_bytes += t.allocated;
        if (t.pack_work) counters.repacked_bytes += t.allocated;
        return &e;
    }

    // Pins experts[0..k) for the largest k that fits next to the other pinned entries. Returns k,
    // or 0 on failure. Only waits for other contexts when nothing could be pinned.
    int acquire(tensor & t, const int32_t * experts, int n, const void ** data, bool count) noexcept {
        std::unique_lock<std::mutex> lock(mutex);
        int k = 0;
        try {
            while (k < n) {
                const int32_t expert = experts[k];
                if (expert < 0 || (size_t) expert >= t.slots.size()) throw std::runtime_error("invalid expert index");
                entry * e = t.slots[expert];
                if (e) {
                    lru.splice(lru.begin(), lru, e->position);
                    if (count) ++counters.hits;
                } else if (make_room(t.allocated)) {
                    e = load(t, expert, count);
                } else if (k > 0) {
                    break; // compute the pinned experts first
                } else {
                    ++waiters;
                    available.wait(lock); // all entries are pinned by other contexts
                    --waiters;
                    continue;
                }
                ++e->pins;
                if (count) touch(t, expert);
                data[k++] = e->data;
            }
            return k;
        } catch (const std::exception & ex) {
            LLAMA_LOG_ERROR("expert cache: %s expert %d: %s\n", t.weight->name, (int) experts[k], ex.what());
        } catch (...) {
            LLAMA_LOG_ERROR("expert cache: %s expert %d: unknown failure\n", t.weight->name, (int) experts[k]);
        }
        unpin(t, experts, k);
        return 0;
    }

    void unpin(tensor & t, const int32_t * experts, int n) {
        for (int i = 0; i < n; ++i) {
            entry * e = t.slots[experts[i]];
            GGML_ASSERT(e && e->pins > 0);
            --e->pins;
        }
        if (waiters) available.notify_all();
    }

    void release(tensor & t, const int32_t * experts, int n) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        unpin(t, experts, n);
    }

    void add_tensor(ggml_tensor * weight, int fd, uint64_t offset) {
        if (!weight || !ggml_is_contiguous(weight) || weight->ne[3] != 1 || weight->ne[2] <= 0 || weight->ne[2] > INT32_MAX) {
            throw std::runtime_error("expert cache: expected a contiguous routed-expert tensor");
        }
        const size_t slice = weight->nb[2];
        size_t pack_work = ggml_cpu_moe_repack_work_size(weight);
        // a budget that only fits a canonical expert stays usable: then skip repacking
        if (GGML_PAD(slice + pack_work, 64) > counters.capacity_bytes) pack_work = 0;
        const size_t allocated = GGML_PAD(slice + pack_work, 64);
        if (allocated > counters.capacity_bytes) {
            throw std::runtime_error("expert cache: tensor " + std::string(weight->name) + " needs " +
                    std::to_string(allocated) + " bytes per expert; increase the expert cache budget");
        }
        struct stat st;
        if (fstat(fd, &st) || st.st_size < 0) throw std::runtime_error("expert cache: cannot stat backing file");
        if (offset > (uint64_t) st.st_size || (uint64_t) weight->ne[2] > ((uint64_t) st.st_size - offset) / slice) {
            throw std::runtime_error("expert cache: expert tensor exceeds backing file");
        }
        std::lock_guard<std::mutex> lock(mutex);
        if (tensors.count(weight)) throw std::runtime_error("expert cache: duplicate tensor registration");
        const auto key = std::make_pair((uint64_t) st.st_dev, (uint64_t) st.st_ino);
        auto f = files.find(key);
        if (f == files.end()) f = files.emplace(key, std::make_unique<file>(fd)).first;
        auto t = std::make_unique<tensor>();
        t->cache     = this;
        t->weight    = weight;
        t->backing   = f->second.get();
        t->offset    = offset;
        t->slice     = slice;
        t->pack_work = pack_work;
        t->allocated = allocated;
        t->slots.resize(weight->ne[2], nullptr);
        t->history.resize(weight->ne[2]);
        t->provider  = { t.get(), acquire_cb, release_cb, prefetch_cb, pack_work != 0 };
        auto & added = *tensors.emplace(weight, std::move(t)).first->second;
        if (!ggml_cpu_register_moe_stream(weight, &added.provider)) {
            tensors.erase(weight);
            throw std::runtime_error("expert cache: too many streamed expert tensors");
        }
        backing_size += (uint64_t) slice * weight->ne[2];
    }

    // ---- usage profile and warm-up ----

    uint64_t fingerprint(const std::vector<tensor *> & ordered) const {
        uint64_t hash = 14695981039346656037ull;
        auto bytes = [&](const void * p, size_t n) {
            for (size_t i = 0; i < n; ++i) { hash ^= static_cast<const unsigned char *>(p)[i]; hash *= 1099511628211ull; }
        };
        for (const auto & f : files) {
            struct stat st;
            if (fstat(f.second->fd, &st)) throw std::runtime_error("cannot fingerprint expert backing file");
#if defined(__APPLE__)
            const int64_t nsec = st.st_mtimespec.tv_nsec;
#else
            const int64_t nsec = st.st_mtim.tv_nsec;
#endif
            const int64_t values[] = { (int64_t) st.st_dev, (int64_t) st.st_ino, (int64_t) st.st_size, (int64_t) st.st_mtime, nsec };
            bytes(values, sizeof(values));
        }
        for (const auto * t : ordered) {
            bytes(t->weight->name, strlen(t->weight->name) + 1);
            bytes(t->weight->ne, sizeof(t->weight->ne));
            const uint64_t values[] = { t->offset, t->slice, (uint64_t) t->weight->type };
            bytes(values, sizeof(values));
        }
        return hash;
    }

    void restore_profile(const std::vector<tensor *> & ordered, size_t n_experts) {
        std::ifstream input(profile_path, std::ios::ate);
        if (!input || input.tellg() < 0 || (uint64_t) input.tellg() > 128 + n_experts * (GGML_MAX_NAME + 100ull)) return;
        input.seekg(0);
        std::string magic;
        uint64_t stored = 0;
        if (!(input >> magic >> std::hex >> stored >> std::dec) || magic != "LCMOE1" || stored != identity) return;
        std::map<std::string, tensor *> named;
        for (auto * t : ordered) named.emplace(t->weight->name, t);
        std::vector<std::tuple<tensor *, int64_t, usage>> restored;
        std::string name;
        int64_t expert;
        usage u;
        while (input >> std::ws && input.peek() != std::char_traits<char>::eof()) {
            if (!(input >> std::quoted(name) >> expert >> u.hits >> u.last)) return;
            const auto found = named.find(name);
            if (restored.size() >= n_experts || found == named.end() || expert < 0 ||
                    (size_t) expert >= found->second->history.size() || u.last == UINT64_MAX) return;
            u.hits = u.hits ? std::max<uint64_t>(1, u.hits / 2) : 0; // decay older workloads
            restored.emplace_back(found->second, expert, u);
        }
        for (const auto & [t, e, saved] : restored) {
            t->history[e] = saved;
            clock = std::max(clock, saved.last);
        }
    }

    bool save_profile() noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (profile_path.empty() || !history_dirty) return true;
            const std::string temporary = profile_path + ".tmp";
            std::ofstream output(temporary, std::ios::trunc);
            output << "LCMOE1 " << std::hex << identity << std::dec << '\n';
            for (const auto & item : tensors) {
                const auto & t = *item.second;
                for (size_t e = 0; e < t.history.size(); ++e) {
                    if (t.history[e].hits) {
                        output << std::quoted(t.weight->name) << ' ' << e << ' ' << t.history[e].hits << ' ' << t.history[e].last << '\n';
                    }
                }
            }
            output.close();
            if (output.fail() || std::rename(temporary.c_str(), profile_path.c_str())) {
                std::remove(temporary.c_str());
                throw std::runtime_error("cannot save expert usage profile " + profile_path);
            }
            history_dirty = false;
            return true;
        } catch (const std::exception & e) {
            LLAMA_LOG_WARN("expert cache: %s\n", e.what());
            return false;
        }
    }

    // load the preferred experts up to the budget: frequent, then recent, else spread over the layers
    bool prefill(const char * path, llama_progress_callback progress, void * user_data) {
        profile_path = path ? path : "";
        std::vector<tensor *> ordered;
        size_t n_experts = 0, max_experts = 0;
        for (const auto & item : tensors) {
            ordered.push_back(item.second.get());
            n_experts  += item.second->slots.size();
            max_experts = std::max(max_experts, item.second->slots.size());
        }
        std::sort(ordered.begin(), ordered.end(), [](const tensor * a, const tensor * b) {
            return strcmp(a->weight->name, b->weight->name) < 0;
        });
        identity = fingerprint(ordered);
        if (!profile_path.empty()) restore_profile(ordered, n_experts);

        struct candidate { tensor * owner; int32_t expert; };
        std::vector<candidate> candidates;
        candidates.reserve(n_experts);
        for (size_t e = 0; e < max_experts; ++e) {
            for (auto * t : ordered) if (e < t->slots.size()) candidates.push_back({ t, (int32_t) e });
        }
        std::stable_sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
            const auto & x = a.owner->history[a.expert];
            const auto & y = b.owner->history[b.expert];
            return x.hits != y.hits ? x.hits > y.hits : x.last > y.last;
        });
        uint64_t total = 0;
        std::vector<candidate> plan;
        for (const auto & c : candidates) {
            if (c.owner->allocated <= counters.capacity_bytes - total) {
                plan.push_back(c);
                total += c.owner->allocated;
            }
        }
        uint64_t done = 0;
        for (const auto & c : plan) {
            if (progress && !progress(total ? (float) done / total : 1.f, user_data)) return false;
            const void * data = nullptr;
            if (acquire(*c.owner, &c.expert, 1, &data, false) != 1) return false;
            release(*c.owner, &c.expert, 1);
            done += c.owner->allocated;
        }
        LLAMA_LOG_INFO("expert cache: prefilled %llu MiB of a %llu MiB budget (%llu MiB of experts)\n",
                (unsigned long long) (counters.used_bytes >> 20), (unsigned long long) (counters.capacity_bytes >> 20),
                (unsigned long long) (backing_size >> 20));
        return !progress || progress(1.f, user_data);
    }
};

llama_moe_stream::llama_moe_stream(uint64_t budget) : pimpl(std::make_unique<impl>(budget)) {}
llama_moe_stream::~llama_moe_stream() = default;

void llama_moe_stream::add_tensor(ggml_tensor * tensor, int fd, uint64_t offset) { pimpl->add_tensor(tensor, fd, offset); }

llama_moe_cache_stats llama_moe_stream::stats() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->counters;
}

uint64_t llama_moe_stream::backing_bytes() const { return pimpl->backing_size; }

bool llama_moe_stream::contains(const ggml_tensor * tensor) const { return pimpl->tensors.count(tensor) != 0; }

bool llama_moe_stream::prefill(const char * path, llama_progress_callback progress, void * user_data) {
    try {
        return pimpl->prefill(path, progress, user_data);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("expert cache warm-up failed: %s\n", e.what());
        return false;
    }
}

bool llama_moe_stream::save_profile() { return pimpl->save_profile(); }

#else

struct llama_moe_stream::impl {};
llama_moe_stream::llama_moe_stream(uint64_t) { throw std::runtime_error("CPU expert streaming requires the CPU backend and POSIX pread"); }
llama_moe_stream::~llama_moe_stream() = default;
void llama_moe_stream::add_tensor(ggml_tensor *, int, uint64_t) {}
llama_moe_cache_stats llama_moe_stream::stats() const { return {}; }
uint64_t llama_moe_stream::backing_bytes() const { return 0; }
bool llama_moe_stream::contains(const ggml_tensor *) const { return false; }
bool llama_moe_stream::prefill(const char *, llama_progress_callback, void *) { return false; }
bool llama_moe_stream::save_profile() { return false; }

#endif
