#include "models.h"

void llama_model_jina_reranker::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen3::load_arch_hparams(ml);

    // the projector is what turns the marker hidden states into ranking embeddings:
    // without it the model would silently serve raw hidden states
    const auto * w_in  = ml.get_weight(tn(LLM_TENSOR_RANK_PROJ_IN,  "weight").str().c_str());
    const auto * w_out = ml.get_weight(tn(LLM_TENSOR_RANK_PROJ_OUT, "weight").str().c_str());
    if (w_in == nullptr || w_out == nullptr) {
        throw std::runtime_error("jina-reranker: the rank projector (rank_proj.in/out) is missing");
    }

    // an embeddings context outputs projected rows
    hparams.n_embd_out_impl = w_out->tensor->ne[1];
}

void llama_model_jina_reranker::load_arch_tensors(llama_model_loader & ml) {
    llama_model_qwen3::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const int64_t n_proj_hidden = ml.get_weight(tn(LLM_TENSOR_RANK_PROJ_IN, "weight").str().c_str())->tensor->ne[1];
    const int64_t n_proj_out    = hparams.n_embd_out();

    rank_proj_in  = create_tensor(tn(LLM_TENSOR_RANK_PROJ_IN,  "weight"), {n_embd,        n_proj_hidden}, 0);
    rank_proj_out = create_tensor(tn(LLM_TENSOR_RANK_PROJ_OUT, "weight"), {n_proj_hidden, n_proj_out},    0);
}

std::unique_ptr<llm_graph_context> llama_model_jina_reranker::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.swa_type == LLAMA_SWA_TYPE_STANDARD) {
        return std::make_unique<graph<true>>(*this, params);
    } else {
        return std::make_unique<graph<false>>(*this, params);
    }
}

template <bool iswa>
llama_model_jina_reranker::graph<iswa>::graph(const llama_model & model_base, const llm_graph_params & params) :
    llama_model_qwen3::graph<iswa>(params, typename llama_model_qwen3::graph<iswa>::no_build()) {
    const auto & model = static_cast<const llama_model_jina_reranker &>(model_base);

    ggml_tensor * hidden = this->build_backbone(model);

    // the LM head, under the same rule as qwen3
    if (!this->cparams.embeddings || this->pooling_type != LLAMA_POOLING_TYPE_NONE) {
        ggml_tensor * logits = this->build_lora_mm(model.output, hidden, model.output_s);

        this->cb(logits, "result_output", -1);
        this->res->t_logits = logits;

        ggml_build_forward_expand(this->gf, logits);
    }

    if (!this->cparams.embeddings) {
        return;
    }

    // rank projector on every output row: [n_embd, n_outputs] -> [n_proj_out, n_outputs]
    ggml_tensor * cur = ggml_mul_mat(this->ctx0, model.rank_proj_in, hidden);
    cur = ggml_relu(this->ctx0, cur);
    this->cb(cur, "rank_proj_hidden", -1);

    cur = ggml_mul_mat(this->ctx0, model.rank_proj_out, cur);
    this->cb(cur, "result_rank_proj", -1);
    this->res->t_embd = cur;

    ggml_build_forward_expand(this->gf, cur);
}

template struct llama_model_jina_reranker::graph<false>;
template struct llama_model_jina_reranker::graph<true>;
