#include "llama.h"
#include "llama-moe-stream.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <future>
#include <list>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(std::string(__func__) + ": " #x); } while (0)

struct temp_file {
    std::string path;
    temp_file() {
        char name[] = "/tmp/test-moe-stream-XXXXXX";
        int fd = mkstemp(name);
        REQUIRE(fd >= 0);
        close(fd);
        path = name;
    }
    ~temp_file() { unlink(path.c_str()); }
};

// pin one expert through the provider, as a MUL_MAT_ID worker would
static const void * pin(const ggml_cpu_moe_stream * p, int32_t expert) {
    const void * data = nullptr;
    return p->acquire(p->context, &expert, 1, &data) == 1 ? data : nullptr;
}

static void unpin(const ggml_cpu_moe_stream * p, int32_t expert) {
    p->release(p->context, &expert, 1);
}

static void test_cache() {
    temp_file file;
    std::vector<float> values(128);
    for (size_t i = 0; i < values.size(); ++i) values[i] = (float) i;
    FILE * f = fopen(file.path.c_str(), "wb+");
    REQUIRE(f);
    REQUIRE(fwrite(values.data(), sizeof(float), values.size(), f) == values.size());
    REQUIRE(fflush(f) == 0);
    ggml_context * ctx = ggml_init({ 2 * ggml_tensor_overhead(), nullptr, true });
    REQUIRE(ctx);
    auto * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 4); // 64 bytes/expert
    auto * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 8, 2); // 128 bytes/expert
    ggml_set_name(a, "small");
    ggml_set_name(b, "large");
    {
        llama_moe_stream cache(192);
        cache.add_tensor(a, fileno(f), 0);
        cache.add_tensor(b, fileno(f), 256);
        REQUIRE(cache.prefill(nullptr));
        REQUIRE(cache.stats().used_bytes == 192);
        const auto * p = ggml_cpu_get_moe_stream(a);
        const auto * q = ggml_cpu_get_moe_stream(b);
        REQUIRE(p && q);
        const void * first = pin(p, 0);
        REQUIRE(first && memcmp(first, values.data(), 64) == 0);
        const void * large = pin(q, 0);
        REQUIRE(large && memcmp(large, values.data() + 64, 128) == 0);
        // a group that cannot pin anything waits for the other context
        auto blocked = std::async(std::launch::async, [&] {
            const void * data = pin(p, 1);
            bool correct = data && memcmp(data, values.data() + 16, 64) == 0;
            unpin(p, 1);
            return correct;
        });
        REQUIRE(blocked.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
        unpin(q, 0);
        REQUIRE(blocked.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        REQUIRE(blocked.get());
        REQUIRE(memcmp(first, values.data(), 64) == 0); // pinned entry survived eviction
        REQUIRE(pin(p, 0) == first);
        unpin(p, 0);
        unpin(p, 0);
        auto st = cache.stats();
        // warm-up reads are not misses
        REQUIRE(st.used_bytes <= 192 && st.misses == 1 && st.hits == 3 && st.bytes_read == 256);
        // a group pins the prefix that fits and leaves the rest for later
        const int32_t group[] = { 0, 1, 2, 3 };
        const void * data[4] = {};
        REQUIRE(p->acquire(p->context, group, 4, data) == 3);
        for (int e = 0; e < 3; ++e) REQUIRE(data[e] && memcmp(data[e], values.data() + 16 * e, 64) == 0);
        p->release(p->context, group, 3);
        REQUIRE(cache.stats().used_bytes == 192);
        p->prefetch(p->context, 3);
        REQUIRE(cache.stats().bytes_read == 256 + 64); // hints cannot fabricate cache fills
        REQUIRE(ftruncate(fileno(f), 64) == 0);
        REQUIRE(pin(p, 3) == nullptr);
        REQUIRE(cache.stats().used_bytes <= 192);
    }
    REQUIRE(!ggml_cpu_get_moe_stream(a));
    bool rejected = false;
    try { llama_moe_stream tiny(63); tiny.add_tensor(a, fileno(f), 0); }
    catch (const std::runtime_error &) { rejected = true; }
    REQUIRE(rejected);
    fclose(f);
    ggml_free(ctx);
    puts("cache: variable-size eviction, pinning, reuse, truncation and undersize passed");
}

static void test_warm_start() {
    temp_file backing, profile;
    FILE * f = fopen(backing.path.c_str(), "wb+");
    REQUIRE(f);
    std::vector<float> values(64);
    for (size_t i = 0; i < values.size(); ++i) values[i] = (float) i;
    REQUIRE(fwrite(values.data(), sizeof(float), values.size(), f) == values.size());
    REQUIRE(fflush(f) == 0);
    auto * ctx = ggml_init({ ggml_tensor_overhead(), nullptr, true });
    REQUIRE(ctx);
    auto * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 4);
    ggml_set_name(t, "warm_experts");
    auto access = [&](llama_moe_stream &, int expert) {
        const auto * p = ggml_cpu_get_moe_stream(t);
        const void * bytes = pin(p, expert);
        REQUIRE(bytes && memcmp(bytes, values.data() + 16 * expert, 64) == 0);
        unpin(p, expert);
    };
    {
        llama_moe_stream cache(128);
        cache.add_tensor(t, fileno(f), 0);
        REQUIRE(cache.prefill(profile.path.c_str()));
        REQUIRE(cache.stats().used_bytes == 128 && cache.stats().bytes_read == 128);
        for (int i = 0; i < 10; ++i) access(cache, 2);
        for (int i = 0; i < 10; ++i) access(cache, 3);
        access(cache, 0); // recent but rare; must not displace the frequent experts on reload
        REQUIRE(cache.save_profile());
    }
    for (uint64_t budget : {64, 100, 128}) {
        // Compare budgets against the same saved workload, without feeding this
        // scenario's extra accesses and frequency decay into the next scenario.
        temp_file reload_profile;
        std::filesystem::copy_file(profile.path, reload_profile.path, std::filesystem::copy_options::overwrite_existing);
        llama_moe_stream cache(budget);
        cache.add_tensor(t, fileno(f), 0);
        REQUIRE(cache.prefill(reload_profile.path.c_str()));
        REQUIRE(cache.stats().used_bytes == (budget / 64) * 64);
        auto before = cache.stats();
        access(cache, 3); // equal-frequency experts: most recent wins
        if (budget == 128) access(cache, 2);
        REQUIRE(cache.stats().misses == before.misses);
        REQUIRE(cache.stats().hits > before.hits);
    }
    REQUIRE(ftruncate(fileno(f), 320) == 0); // backing identity changed: discard previous preferences
    {
        llama_moe_stream cache(64);
        cache.add_tensor(t, fileno(f), 0);
        REQUIRE(cache.prefill(profile.path.c_str()));
        auto before = cache.stats();
        access(cache, 0);
        REQUIRE(cache.stats().misses == before.misses);
        access(cache, 3);
        REQUIRE(cache.stats().misses == before.misses + 1);
    }
    { std::ofstream corrupt(profile.path); corrupt << "invalid profile\n"; }
    {
        llama_moe_stream cache(1024);
        cache.add_tensor(t, fileno(f), 0);
        REQUIRE(cache.prefill(profile.path.c_str()));
        REQUIRE(cache.stats().used_bytes == 256); // no allocation beyond the expert set
        REQUIRE(cache.stats().bytes_read == 256 && cache.stats().misses == 0);
    }
    {
        llama_moe_stream cache(128);
        cache.add_tensor(t, fileno(f), 0);
        int callbacks = 0;
        REQUIRE(!cache.prefill(nullptr, [](float, void * user) { return ++*static_cast<int *>(user) < 2; }, &callbacks));
        REQUIRE(cache.stats().used_bytes == 64);
        access(cache, 3); // cancellation released the warm-up lease
    }
    {
        llama_moe_stream cache(128);
        cache.add_tensor(t, fileno(f), 0);
        REQUIRE(ftruncate(fileno(f), 0) == 0);
        REQUIRE(!cache.prefill(nullptr));
        REQUIRE(cache.stats().used_bytes == 0);
    }
    ggml_free(ctx);
    fclose(f);
    puts("warm start: budget fill, hot/recent ordering, reload, resized budgets, identity, corruption, cancellation and I/O failure passed");
}

static void test_repacked_experts() {
    temp_file file;
    auto * ctx = ggml_init({ 1024 * 1024, nullptr, false });
    REQUIRE(ctx);
    auto * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, 256, 64, 4);
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, 40);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 40);
    std::vector<float> raw(ggml_nelements(weights));
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = 0.03f * sinf(i * 0.17f);
    REQUIRE(ggml_quantize_chunk(weights->type, raw.data(), weights->data, 0, 256, 256, nullptr) == ggml_nbytes(weights));
    for (int i = 0; i < ggml_nelements(input); ++i) ((float *) input->data)[i] = sinf(i * 0.13f);
    for (int t = 0; t < 40; ++t) {
        ((int32_t *) ids->data)[2*t] = t % 4;
        ((int32_t *) ids->data)[2*t+1] = t % 7 == 0 ? -1 : (t+1) % 4;
    }
    FILE * f = fopen(file.path.c_str(), "wb+");
    REQUIRE(f && fwrite(weights->data, 1, ggml_nbytes(weights), f) == ggml_nbytes(weights));
    REQUIRE(fflush(f) == 0);
    auto * out = ggml_mul_mat_id(ctx, weights, input, ids);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    std::vector<uint8_t> work;
    auto compute = [&](int threads) {
        auto plan = ggml_graph_plan(graph, threads, nullptr);
        work.resize(plan.work_size);
        plan.work_data = work.data();
        return ggml_graph_compute(graph, &plan);
    };
    REQUIRE(compute(1) == GGML_STATUS_SUCCESS);
    const auto * values = (const float *) out->data;
    std::vector<float> reference(values, values + ggml_nelements(out));
    const size_t scratch = ggml_cpu_moe_repack_work_size(weights);
    const size_t budget = (weights->nb[2] + scratch + 63) & ~(size_t) 63;
    {
        llama_moe_stream warm(4 * budget);
        warm.add_tensor(weights, fileno(f), 0);
        REQUIRE(warm.prefill(nullptr));
        const auto before = warm.stats();
        REQUIRE(compute(4) == GGML_STATUS_SUCCESS);
        for (size_t i = 0; i < reference.size(); ++i) REQUIRE(fabsf(values[i] - reference[i]) < 1e-4f);
        const auto after = warm.stats();
        REQUIRE(after.hits == before.hits + 4 && after.misses == before.misses && after.bytes_read == before.bytes_read);
    }
    {
        llama_moe_stream cache(budget); // only one expert, even when the selected union is four
        cache.add_tensor(weights, fileno(f), 0);
        for (int threads : { 1, 4, 16 }) {
            REQUIRE(compute(threads) == GGML_STATUS_SUCCESS);
            for (size_t i = 0; i < reference.size(); ++i) REQUIRE(fabsf(values[i] - reference[i]) < 1e-4f);
            auto st = cache.stats();
            REQUIRE(st.used_bytes <= budget);
            REQUIRE(st.repacked_bytes == (scratch ? st.used_bytes : 0));
        }
        // A failed acquisition must terminate all workers, not strand them at a barrier.
        REQUIRE(ftruncate(fileno(f), 0) == 0);
        REQUIRE(compute(16) == GGML_STATUS_FAILED);
    }
    fclose(f);
    ggml_free(ctx);
    printf("Q4_K experts: %s, one-expert budget, 1/4/16 workers, masked IDs and I/O failure passed\n",
            scratch ? "CPU_REPACK" : "canonical kernels (no repack ISA)");
}

static constexpr int EMBED = 64, FF = 96, VOCAB = 260, EXPERTS = 4, LAYERS = 2;
static constexpr uint64_t SLICE = EMBED * FF * sizeof(float);

static void write_model(const std::string & path, bool quantized = false) {
    gguf_context * g = gguf_init_empty();
    ggml_context * ctx = ggml_init({ 64 * ggml_tensor_overhead(), nullptr, true });
    REQUIRE(g && ctx);
    gguf_set_val_str(g, "general.architecture", "qwen3moe");
    gguf_set_val_str(g, "general.name", "CPU expert-streaming test fixture");
    auto u32 = [&](const char * key, uint32_t value) { gguf_set_val_u32(g, (std::string("qwen3moe.") + key).c_str(), value); };
    u32("context_length", 512); u32("embedding_length", EMBED); u32("block_count", LAYERS);
    u32("feed_forward_length", FF); u32("expert_feed_forward_length", FF);
    u32("expert_count", EXPERTS); u32("expert_used_count", 2);
    u32("attention.head_count", 2); u32("attention.head_count_kv", 2); u32("rope.dimension_count", 32);
    gguf_set_val_f32(g, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_f32(g, "qwen3moe.rope.freq_base", 10000.f);
    std::vector<std::string> tokens = { "<unk>", "<s>", "</s>", "\xe2\x96\x81" };
    std::vector<int32_t> types = { 2, 3, 3, 1 };
    for (int i = 0; i < 256; ++i) {
        char token[16]; snprintf(token, sizeof(token), "<0x%02X>", i);
        tokens.emplace_back(token); types.push_back(6);
    }
    std::vector<const char *> pieces;
    for (const auto & token : tokens) pieces.push_back(token.c_str());
    std::vector<float> scores(VOCAB, 0.f);
    gguf_set_val_str(g, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", pieces.data(), pieces.size());
    gguf_set_arr_data(g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_u32(g, "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_str(g, "tokenizer.chat_template", "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}");
    std::list<std::vector<float>> payloads; // GGUF retains pointers until write_to_file
    std::list<std::vector<uint8_t>> quantized_payloads;
    std::mt19937 random(42);
    std::normal_distribution<float> dist(0.f, 0.03f);
    auto add = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        const auto type = quantized && name.find("_exps.") != std::string::npos ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
        ggml_tensor * t = ggml_new_tensor(ctx, type, shape.size(), shape.begin());
        ggml_set_name(t, name.c_str());
        payloads.emplace_back(ggml_nelements(t));
        for (auto & v : payloads.back()) v = name.find("norm") != std::string::npos ? 1.f : dist(random);
        t->data = payloads.back().data();
        if (type != GGML_TYPE_F32) {
            quantized_payloads.emplace_back(ggml_nbytes(t));
            REQUIRE(ggml_quantize_chunk(type, payloads.back().data(), quantized_payloads.back().data(), 0,
                    ggml_nelements(t) / t->ne[0], t->ne[0], nullptr) == ggml_nbytes(t));
            t->data = quantized_payloads.back().data();
        }
        gguf_add_tensor(g, t);
    };
    add("token_embd.weight", { EMBED, VOCAB });
    add("output_norm.weight", { EMBED });
    add("output.weight", { EMBED, VOCAB });
    for (int layer = 0; layer < LAYERS; ++layer) {
        std::string b = "blk." + std::to_string(layer) + ".";
        add(b + "attn_norm.weight", { EMBED });
        for (const char * name : { "attn_q", "attn_k", "attn_v", "attn_output" }) add(b + name + ".weight", { EMBED, EMBED });
        add(b + "attn_q_norm.weight", { 32 }); add(b + "attn_k_norm.weight", { 32 });
        add(b + "ffn_norm.weight", { EMBED });
        add(b + "ffn_gate_inp.weight", { EMBED, EXPERTS });
        add(b + "ffn_gate_exps.weight", { EMBED, FF, EXPERTS });
        add(b + "ffn_down_exps.weight", { FF, EMBED, EXPERTS });
        add(b + "ffn_up_exps.weight", { EMBED, FF, EXPERTS });
    }
    REQUIRE(gguf_write_to_file(g, path.c_str(), false));
    gguf_free(g);
    ggml_free(ctx);
}

static llama_model * load(const std::string & path, uint64_t budget) {
    auto p = llama_model_default_params();
    p.n_gpu_layers = 0;
    p.use_extra_bufts = false;
    p.moe_cache_bytes = budget;
    return llama_model_load_from_file(path.c_str(), p);
}

static llama_context * context(llama_model * model, int threads) {
    auto p = llama_context_default_params();
    p.n_ctx = 512; p.n_batch = 64; p.n_ubatch = 64;
    p.n_threads = threads; p.n_threads_batch = threads;
    auto * ctx = llama_init_from_model(model, p);
    REQUIRE(ctx);
    return ctx;
}

static std::vector<float> generate(llama_model * model, int threads) {
    auto * ctx = context(model, threads);
    std::vector<llama_token> prompt = { 1, 104, 105, 106, 120, 121, 100, 110 };
    REQUIRE(llama_decode(ctx, llama_batch_get_one(prompt.data(), prompt.size())) == 0);
    std::vector<float> result;
    for (int i = 0; i < 5; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        REQUIRE(logits);
        result.insert(result.end(), logits, logits + VOCAB);
        llama_token next = std::max_element(logits, logits + VOCAB) - logits;
        REQUIRE(llama_decode(ctx, llama_batch_get_one(&next, 1)) == 0);
    }
    llama_free(ctx);
    return result;
}

static void equal(const std::vector<float> & a, const std::vector<float> & b) {
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) REQUIRE(std::isfinite(a[i]) && std::isfinite(b[i]) && std::abs(a[i] - b[i]) < 1e-5f);
}

static void test_model(const std::string & path, uint64_t slice = SLICE) {
    auto * normal = load(path, 0); REQUIRE(normal);
    auto reference = generate(normal, 4);
    llama_model_free(normal);
    auto * low = load(path, slice); REQUIRE(low);
    equal(reference, generate(low, 4)); // two selected experts exceed the one-slice budget
    llama_moe_cache_stats st;
    REQUIRE(llama_model_get_moe_cache_stats(low, &st));
    REQUIRE(st.used_bytes == slice && st.capacity_bytes == slice && st.misses > 2 && st.bytes_read > slice);
    llama_model_free(low);
    const uint64_t all = 3 * LAYERS * EXPERTS * slice;
    auto * resident = load(path, all); REQUIRE(resident); // everything fits: loaded normally
    REQUIRE(llama_model_moe_cache_prefill(resident, nullptr, nullptr, nullptr));
    equal(reference, generate(resident, 4));
    REQUIRE(llama_model_get_moe_cache_stats(resident, &st));
    REQUIRE(st.capacity_bytes == all && st.used_bytes == all && st.bytes_read == 0 && st.hits == 0);
    llama_model_free(resident);
    auto * warm = load(path, all - 1); REQUIRE(warm); // streamed, nearly everything warm
    temp_file profile;
    REQUIRE(llama_model_moe_cache_prefill(warm, profile.path.c_str(), nullptr, nullptr));
    REQUIRE(llama_model_get_moe_cache_stats(warm, &st));
    REQUIRE(st.used_bytes >= all / 2 && st.used_bytes <= st.capacity_bytes && st.misses == 0);
    equal(reference, generate(warm, 1));
    REQUIRE(llama_model_get_moe_cache_stats(warm, &st));
    ggml_cpu_set_moe_prefetch(true);
    equal(reference, generate(warm, 4));
    ggml_cpu_set_moe_prefetch(false);
    REQUIRE(llama_model_get_moe_cache_stats(warm, &st));
    REQUIRE(st.hits > 0 && st.used_bytes <= st.capacity_bytes);
    llama_model_free(warm);
    REQUIRE(load(path, slice - 1) == nullptr);

    temp_file truncated;
    std::filesystem::copy_file(path, truncated.path, std::filesystem::copy_options::overwrite_existing);
    auto * broken = load(truncated.path, slice); REQUIRE(broken);
    auto * ctx = context(broken, 4);
    // Remove the final expert tensor only: ordinary mapped weights remain valid.
    auto size = std::filesystem::file_size(truncated.path);
    REQUIRE(truncate(truncated.path.c_str(), size - EXPERTS * slice) == 0);
    llama_token prompt[] = { 1, 104, 105, 106 };
    REQUIRE(llama_decode(ctx, llama_batch_get_one(prompt, 4)) == -3);
    llama_free(ctx);
    llama_model_free(broken);
    puts("model: prompt/decode logits, single/four threads, union > budget, warm reuse, prefetch and I/O failure passed");
}

int main(int argc, char ** argv) {
    try {
        if (argc == 3 && !strcmp(argv[1], "--write-model")) { write_model(argv[2]); return 0; }
        REQUIRE(argc == 1);
        llama_backend_init();
        test_cache();
        test_warm_start();
        test_repacked_experts();
        temp_file model;
        write_model(model.path);
        test_model(model.path);
        write_model(model.path, true);
        test_model(model.path, ggml_row_size(GGML_TYPE_Q8_0, EMBED) * FF);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
}
