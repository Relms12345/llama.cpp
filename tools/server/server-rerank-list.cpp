#include "server-rerank-list.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// jina's eval driver (rerank.py): limits, flush rules and prompt
static const int64_t RERANK_LIST_MAX_LENGTH     = 131072; // tokenizer model_max_length
static const int64_t RERANK_LIST_MAX_DOC        = 8192;
static const int64_t RERANK_LIST_DOC_RESERVE    = 1;
static const int64_t RERANK_LIST_MAX_QUERY      = 2048;
static const int64_t RERANK_LIST_QUERY_RESERVE  = 64;     // prompt scaffolding
static const size_t  RERANK_LIST_BLOCK_SIZE     = 125;
static const double  RERANK_LIST_EPS            = 1e-8;

static const char * RERANK_LIST_DOC_TOKEN   = "<|embed_token|>";
static const char * RERANK_LIST_QUERY_TOKEN = "<|rerank_token|>";
static const char * RERANK_LIST_SCORE_TOKEN = "<|score_token|>";

static const char * RERANK_LIST_SYSTEM =
    "<|im_start|>system\n"
    "You are a search relevance expert who can determine a ranking of the passages "
    "based on how relevant they are to the query. If the query is a question, how "
    "relevant a passage is depends on how well it answers the question. If not, try "
    "to analyze the intent of the query and assess how well each passage satisfies "
    "the intent. If an instruction is provided, you should follow the instruction "
    "when determining the ranking."
    "<|im_end|>\n<|im_start|>user\n";

static const char * RERANK_LIST_RANKING =
    "\nPlease provide the ranking of all passages based on their relevance "
    "to the search query, in descending order of relevance, with each label "
    "enclosed in square brackets (e.g., [2] > [1] > [3] > [0]).";

static const char * RERANK_LIST_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

// like Python's bytes.decode(errors="replace") as used by the HF tokenizer:
// each maximal invalid subsequence becomes one U+FFFD
static std::string rerank_list_utf8_lossy(const std::string & s) {
    static const char * REPLACEMENT = "\xEF\xBF\xBD";

    std::string out;
    out.reserve(s.size());

    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = s[i];
        if (c < 0x80) {
            out += (char) c;
            i++;
            continue;
        }

        size_t n = 0; // continuation bytes
        unsigned char lo = 0x80;
        unsigned char hi = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) {
            n = 1;
        } else if (c >= 0xE0 && c <= 0xEF) {
            n  = 2;
            lo = c == 0xE0 ? 0xA0 : 0x80;
            hi = c == 0xED ? 0x9F : 0xBF;
        } else if (c >= 0xF0 && c <= 0xF4) {
            n  = 3;
            lo = c == 0xF0 ? 0x90 : 0x80;
            hi = c == 0xF4 ? 0x8F : 0xBF;
        } else {
            out += REPLACEMENT;
            i++;
            continue;
        }

        size_t j = 1;
        for (; j <= n && i + j < s.size(); j++) {
            const unsigned char cc = s[i + j];
            if (cc < (j == 1 ? lo : 0x80) || cc > (j == 1 ? hi : 0xBF)) {
                break;
            }
        }
        if (j == n + 1) {
            out.append(s, i, n + 1);
        } else {
            out += REPLACEMENT;
        }
        i += j;
    }

    return out;
}

static std::string rerank_list_sanitize(std::string text) {
    string_replace_all(text, RERANK_LIST_DOC_TOKEN,   "");
    string_replace_all(text, RERANK_LIST_QUERY_TOKEN, "");
    string_replace_all(text, RERANK_LIST_SCORE_TOKEN, "");
    return text;
}

static std::string rerank_list_prompt(
        const std::string & query,
        const std::vector<std::string> & docs,
        const std::string & instruction) {
    const std::string q = rerank_list_sanitize(query);

    std::string body = "I will provide you with " + std::to_string(docs.size()) + " passages, each indicated by a numerical identifier. "
                       "Rank the passages based on their relevance to query: " + q + RERANK_LIST_QUERY_TOKEN + "\n";
    if (!instruction.empty()) {
        body += "<instruct>\n" + instruction + "\n</instruct>\n";
    }
    for (size_t i = 0; i < docs.size(); i++) {
        if (i > 0) {
            body += "\n";
        }
        body += "<passage id=\"" + std::to_string(i) + "\">\n" + rerank_list_sanitize(docs[i]) + RERANK_LIST_DOC_TOKEN + "\n</passage>";
    }
    body += "\n<query>\n" + q + RERANK_LIST_QUERY_TOKEN + "\n</query>";
    body += RERANK_LIST_RANKING;

    return RERANK_LIST_SYSTEM + body + RERANK_LIST_SUFFIX;
}

void server_rerank_list_context::init(const llama_model * model, const llama_context * ctx) {
    *this = server_rerank_list_context(); // the model can be reloaded

    char arch[64];
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0 || std::string(arch) != "jina-reranker") {
        return;
    }

    vocab = llama_model_get_vocab(model);

    auto single_token = [&](const char * text) {
        const llama_tokens ids = common_tokenize(vocab, text, false, true);
        if (ids.size() != 1) {
            throw std::runtime_error(string_format("listwise reranker: %s is not one token", text));
        }
        return ids[0];
    };
    token_doc   = single_token(RERANK_LIST_DOC_TOKEN);
    token_query = single_token(RERANK_LIST_QUERY_TOKEN);

    unpooled = llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_NONE;
    enabled  = true;
}

std::string server_rerank_list_context::truncate(const std::string & text, size_t max_tokens, size_t & n_tokens) const {
    llama_tokens ids = common_tokenize(vocab, text, false, true);
    if (ids.size() < max_tokens) {
        n_tokens = ids.size();
        return text;
    }
    ids.resize(max_tokens);
    n_tokens = max_tokens;
    return rerank_list_utf8_lossy(common_detokenize(vocab, ids, false));
}

std::vector<server_rerank_list_block> server_rerank_list_context::make_blocks(
        const std::string & query,
        const std::vector<std::string> & documents,
        const std::string & instruction) const {
    GGML_ASSERT(enabled);

    std::vector<std::string> docs(documents.size());
    std::vector<int64_t>     lengths(documents.size());
    for (size_t i = 0; i < documents.size(); i++) {
        size_t n = 0;
        docs[i]    = truncate(documents[i], RERANK_LIST_MAX_DOC - RERANK_LIST_DOC_RESERVE, n);
        lengths[i] = (int64_t) n;
    }

    size_t n_query = 0;
    const std::string query_trunc = truncate(query, RERANK_LIST_MAX_QUERY - RERANK_LIST_QUERY_RESERVE, n_query);

    // flush rules of the eval driver
    const int64_t capacity_base = RERANK_LIST_MAX_LENGTH - (int64_t) n_query;
    int64_t capacity = capacity_base;
    int64_t n_cur    = (int64_t) n_query;

    std::vector<server_rerank_list_block> blocks;
    server_rerank_list_block cur;
    for (size_t i = 0; i < docs.size(); i++) {
        cur.docs.push_back(i);
        capacity -= lengths[i];
        n_cur    += lengths[i];

        if (cur.docs.size() >= RERANK_LIST_BLOCK_SIZE ||
            capacity < RERANK_LIST_MAX_DOC ||
            n_cur >= RERANK_LIST_MAX_LENGTH - RERANK_LIST_MAX_DOC) {
            blocks.push_back(std::move(cur));
            cur      = server_rerank_list_block();
            capacity = capacity_base;
            n_cur    = (int64_t) n_query;
        }
    }
    if (!cur.docs.empty()) {
        blocks.push_back(std::move(cur));
    }

    for (auto & block : blocks) {
        std::vector<std::string> block_docs;
        for (const size_t i : block.docs) {
            block_docs.push_back(docs[i]);
        }
        block.prompt = rerank_list_prompt(query_trunc, block_docs, instruction);
        block.tokens = common_tokenize(vocab, block.prompt, false, true);

        for (size_t pos = 0; pos < block.tokens.size(); pos++) {
            if (block.tokens[pos] == token_doc || block.tokens[pos] == token_query) {
                block.markers.push_back((int32_t) pos);
            }
        }

        // early query, one marker per document, late query
        bool ok = block.markers.size() == block.docs.size() + 2;
        for (size_t m = 0; ok && m < block.markers.size(); m++) {
            const bool is_query = m == 0 || m + 1 == block.markers.size();
            ok = block.tokens[block.markers[m]] == (is_query ? token_query : token_doc);
        }
        if (!ok) {
            throw std::invalid_argument("the instruction must not contain the marker tokens of the model");
        }
    }

    return blocks;
}

// cosine with the epsilon of the eval driver in the denominator
template <typename A, typename B>
static double rerank_list_cosine(const A * a, const B * b, size_t n) {
    double dot = 0.0;
    double na  = 0.0;
    double nb  = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + RERANK_LIST_EPS);
}

std::vector<double> server_rerank_list_context::score(
        const std::vector<server_rerank_list_block> & blocks,
        const std::vector<std::vector<std::vector<float>>> & vectors,
        size_t n_documents,
        std::vector<double> & weights) {
    GGML_ASSERT(blocks.size() == vectors.size() && !blocks.empty());
    const size_t n_embd = vectors[0][0].size();

    // weight of a block: its best rescaled cosine, documents vs the late query
    weights.assign(blocks.size(), 0.0);
    std::vector<double> fused(n_embd, 0.0);
    double weight_sum = 0.0;
    for (size_t b = 0; b < blocks.size(); b++) {
        GGML_ASSERT(vectors[b].size() == blocks[b].docs.size() + 2);
        const auto & query = vectors[b].back();
        double w = 0.0;
        for (size_t d = 0; d < blocks[b].docs.size(); d++) {
            const double c = rerank_list_cosine(vectors[b][1 + d].data(), query.data(), n_embd);
            w = d == 0 ? (1.0 + c) / 2.0 : std::max(w, (1.0 + c) / 2.0);
        }
        weights[b] = w;
        weight_sum += w;
        for (size_t k = 0; k < n_embd; k++) {
            fused[k] += w * (double) query[k];
        }
    }
    for (auto & v : fused) {
        v /= weight_sum;
    }

    std::vector<double> scores(n_documents, 0.0);
    for (size_t b = 0; b < blocks.size(); b++) {
        for (size_t d = 0; d < blocks[b].docs.size(); d++) {
            scores[blocks[b].docs[d]] = rerank_list_cosine(vectors[b][1 + d].data(), fused.data(), n_embd);
        }
    }
    return scores;
}
