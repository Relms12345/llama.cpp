#include "models.h"

void llama_model_jina_reranker::load_arch_tensors(llama_model_loader & ml) {
    llama_model_qwen3::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    // the projector is what turns the marker hidden states into ranking embeddings:
    // without it the model would silently serve raw hidden states
    const auto * w_in  = ml.get_weight(tn(LLM_TENSOR_RANK_PROJ_IN,  "weight").str().c_str());
    const auto * w_out = ml.get_weight(tn(LLM_TENSOR_RANK_PROJ_OUT, "weight").str().c_str());
    if (w_in == nullptr || w_out == nullptr) {
        throw std::runtime_error("jina-reranker: the rank projector (rank_proj.in/out) is missing");
    }
    const int64_t n_proj_hidden = w_in->tensor->ne[1];
    const int64_t n_proj_out    = w_out->tensor->ne[1];

    rank_proj_in  = create_tensor(tn(LLM_TENSOR_RANK_PROJ_IN,  "weight"), {n_embd,        n_proj_hidden}, 0);
    rank_proj_out = create_tensor(tn(LLM_TENSOR_RANK_PROJ_OUT, "weight"), {n_proj_hidden, n_proj_out},    0);
}
