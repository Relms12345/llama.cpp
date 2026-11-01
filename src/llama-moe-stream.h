#pragma once

#include "llama.h"

#include <cstdint>
#include <memory>

struct ggml_tensor;

// Model-owned bounded cache of routed-expert slices, read from the GGUF file on demand.
// Contexts must be destroyed before their model. Payloads are bounded; metadata and OS page cache are not.
class LLAMA_API llama_moe_stream {
public:
    explicit llama_moe_stream(uint64_t budget);
    ~llama_moe_stream();
    llama_moe_stream(const llama_moe_stream &) = delete;
    llama_moe_stream & operator=(const llama_moe_stream &) = delete;

    void add_tensor(ggml_tensor * tensor, int fd, uint64_t offset);
    llama_moe_cache_stats stats() const;
    uint64_t backing_bytes() const;
    bool contains(const ggml_tensor * tensor) const;
    bool prefill(const char * profile_path, llama_progress_callback progress = nullptr, void * user_data = nullptr);
    bool save_profile();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
