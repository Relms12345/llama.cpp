#pragma once

#include "server-common.h"

#include <string>
#include <vector>

// listwise rerankers (jina-reranker-v3.5): the documents of a request share prompts ("blocks"),
// the model writes one vector at each document marker and at each query marker, and the score
// of a document is the cosine of its vector and the query vector fused over the blocks
// the truncation, packing, prompt and fusion follow jina's eval driver (rerank.py of the GGUF repo)

struct server_rerank_list_block {
    std::vector<size_t>  docs;    // indices into the request documents, in order
    std::string          prompt;
    llama_tokens         tokens;
    std::vector<int32_t> markers; // prompt positions of the vectors: early query, one per document, late query
};

struct server_rerank_list_context {
    bool   enabled    = false;
    bool   unpooled   = false; // the context has no pooling, so the vectors can be read per token
    size_t block_size = 0;     // max documents per block (jina: 125)

    // enabled for the jina-reranker arch, throws if the vocab lacks its marker tokens or block_size is not 1 to 125
    void init(const llama_model * model, const llama_context * ctx, int32_t block_size);

    // truncate the query and the documents, pack the documents into blocks, render and tokenize their prompts
    // throws std::invalid_argument if a prompt does not carry exactly its markers
    std::vector<server_rerank_list_block> make_blocks(
            const std::string & query,
            const std::vector<std::string> & documents,
            const std::string & instruction) const;

    // vectors[b]: the vectors of block b at its markers, in order
    // returns the score of every request document; weights receives the fusion weight of each block
    static std::vector<double> score(
            const std::vector<server_rerank_list_block> & blocks,
            const std::vector<std::vector<std::vector<float>>> & vectors,
            size_t n_documents,
            std::vector<double> & weights);

private:
    const llama_vocab * vocab = nullptr;
    llama_token token_doc   = LLAMA_TOKEN_NULL;
    llama_token token_query = LLAMA_TOKEN_NULL;

    // the first max_tokens tokens of text, decoded without the control tokens, if text has at least that many
    std::string truncate(const std::string & text, size_t max_tokens, size_t & n_tokens) const;
};
