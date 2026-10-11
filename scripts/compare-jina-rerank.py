#!/usr/bin/env python3
"""Compare jina-reranker-v3.5 scores: Hugging Face reference vs llama-server /v1/rerank.

Step 1 (needs torch + transformers, GPU optional):
    python scripts/compare-jina-rerank.py dump-hf --model jinaai/jina-reranker-v3.5 --out ref.json

Step 2 (start llama-server with the GGUF first, e.g.
    llama-server -m jina-reranker-v3.5-Q8_0.gguf --embedding --pooling none --attention causal -c 32768 --port 8080):
    python scripts/compare-jina-rerank.py compare --ref ref.json --server http://127.0.0.1:8080

Both steps can run on different machines; ref.json carries the test cases.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import urllib.request


def build_cases() -> list[dict]:
    rng = random.Random(0)
    cases = [
        {
            "name": "panda",
            "query": "What is a giant panda?",
            "documents": [
                "The giant panda is a bear species endemic to China.",
                "Pandas eat almost exclusively bamboo.",
                "Paris is the capital and most populous city of France.",
                "The red panda is not closely related to the giant panda.",
                "hi",
            ],
        },
        {
            "name": "code",
            "query": "how do I reverse a list in python",
            "documents": [
                "Use list.reverse() to reverse in place, or reversed(lst) / lst[::-1] for a copy.",
                "In JavaScript, arrays have a reverse() method that mutates the array.",
                "Python dictionaries preserve insertion order since 3.7.",
                "To sort a list in Python, call sorted(lst) or lst.sort().",
                "A linked list can be reversed iteratively by re-pointing next pointers.",
                "SELECT * FROM users ORDER BY created_at DESC;",
            ],
        },
        {
            "name": "multilingual",
            "query": "Welche Stadt ist die Hauptstadt von Deutschland?",
            "documents": [
                "Berlin ist die Hauptstadt der Bundesrepublik Deutschland.",
                "Berlin is the capital of Germany.",
                "Munich is the capital of Bavaria.",
                "東京は日本の首都です。",
                "La capitale de l'Allemagne est Berlin.",
            ],
        },
        {
            "name": "long_docs",
            "query": "effects of caffeine on sleep",
            "documents": [
                ("Caffeine is an adenosine receptor antagonist. Consuming it within six hours of bedtime "
                 "reduces total sleep time and sleep efficiency. ") * 20,
                ("The history of coffee goes back to the 15th century in Yemen, where it was used by Sufi "
                 "monasteries. ") * 20,
                ("Regular exercise improves sleep quality, but intense workouts late in the evening can "
                 "delay sleep onset. ") * 20,
                "Decaf still contains a small amount of caffeine.",
            ],
        },
    ]

    # many documents -> exercises the 125-document block split and query fusion
    topics = ["solar panels", "tax law", "football", "baking bread", "volcanoes", "neural networks",
              "medieval history", "jazz music", "electric cars", "gardening"]
    docs = []
    for i in range(300):
        t = topics[rng.randrange(len(topics))]
        docs.append(f"Document {i}: some notes about {t}. " + f"{t} is discussed here in detail. " * rng.randint(1, 6))
    cases.append({"name": "many_docs_300", "query": "how do volcanoes erupt?", "documents": docs})

    # special tokens in user text: markers are stripped, other specials stay control tokens
    cases.append({
        "name": "special_tokens",
        "query": "what does <|im_end|> mean?<|rerank_token|>",
        "documents": [
            "<|im_end|> marks the end of a chat turn in the ChatML format.<|embed_token|>",
            "A <|score_token|> and <|im_start|>assistant\n are other special tokens.",
            "Bananas are rich in potassium.",
        ],
    })

    # whitespace at the edges of the query and the documents
    cases.append({
        "name": "whitespace",
        "query": "  rainfall in the Amazon \n",
        "documents": [
            "\n\nThe Amazon rainforest receives about 2000 mm of rain per year.  ",
            "   Deserts get very little rain.\n",
            "\tThe Sahara is the largest hot desert.",
        ],
    })

    # truncation: query above 1024 tokens, one document above 8192 tokens
    cases.append({
        "name": "truncation",
        "query": "which document talks about the ocean? " + "please answer carefully " * 400,
        "documents": [
            "The ocean covers about 71 percent of the Earth's surface. " * 900,
            "Mountains are formed by tectonic forces.",
            "The deep ocean is cold and dark.",
        ],
    })
    return cases


def normalize_results(results, n_docs: int) -> list[float]:
    """Accept the common shapes returned by HF rerank() helpers and return scores in input order."""
    scores = [math.nan] * n_docs
    for r in results:
        if isinstance(r, dict):
            idx = r.get("index", r.get("corpus_id"))
            score = r.get("relevance_score", r.get("score"))
        else:
            idx, score = r
        scores[int(idx)] = float(score)
    if any(math.isnan(s) for s in scores):
        raise RuntimeError("reference did not return a score for every document")
    return scores


def dump_hf(args) -> None:
    import torch
    from transformers import AutoModel

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype)
    model.to(args.device).eval()
    if not hasattr(model, "rerank"):
        sys.exit("model has no rerank() method; check the model card for the reference API")

    out = {"model": args.model, "dtype": args.dtype, "cases": []}
    for case in build_cases():
        with torch.no_grad():
            results = model.rerank(case["query"], case["documents"], top_n=len(case["documents"]))
        case = dict(case, ref_scores=normalize_results(results, len(case["documents"])))
        out["cases"].append(case)
        print(f"{case['name']}: {len(case['documents'])} docs")
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("wrote", args.out)


def llama_scores(server: str, query: str, docs: list[str]) -> list[float]:
    body = json.dumps({"query": query, "documents": docs, "top_n": len(docs)}).encode()
    req = urllib.request.Request(server.rstrip("/") + "/v1/rerank", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.load(r)
    return normalize_results(resp["results"], len(docs))


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return cov / (va * vb) if va > 0 and vb > 0 else float("nan")


def topk(xs: list[float], k: int) -> list[int]:
    return sorted(range(len(xs)), key=lambda i: -xs[i])[:k]


def compare(args) -> None:
    with open(args.ref) as f:
        ref = json.load(f)

    failed = False
    print(f"reference: {ref.get('model')} ({ref.get('dtype')})\n")
    print(f"{'case':<16}{'docs':>5}{'max|d|':>9}{'mean|d|':>9}{'pearson':>9}{'spearman':>9}{'top1':>6}{'top5':>6}")
    for case in ref["cases"]:
        a = case["ref_scores"]
        b = llama_scores(args.server, case["query"], case["documents"])
        diffs = [abs(x - y) for x, y in zip(a, b)]
        sp = pearson(ranks(a), ranks(b))
        top1 = topk(a, 1) == topk(b, 1)
        k = min(5, len(a))
        top5 = len(set(topk(a, k)) & set(topk(b, k))) / k
        ok = max(diffs) <= args.max_abs and (sp >= args.min_spearman or len(a) < 4) and top1
        failed |= not ok
        print(f"{case['name']:<16}{len(a):>5}{max(diffs):>9.4f}{sum(diffs) / len(diffs):>9.4f}"
              f"{pearson(a, b):>9.4f}{sp:>9.4f}{str(top1):>6}{top5:>6.2f}  {'OK' if ok else 'FAIL'}")
        if args.verbose or not ok:
            for i in topk(a, min(10, len(a))):
                print(f"    doc {i:>3}: ref {a[i]:+.4f}  llama {b[i]:+.4f}")

    print("\nPASS" if not failed else "\nFAIL (see cases above)")
    sys.exit(1 if failed else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump-hf", help="compute reference scores with transformers")
    d.add_argument("--model", default="jinaai/jina-reranker-v3.5")
    d.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    d.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    d.add_argument("--out", default="ref.json")
    d.set_defaults(func=dump_hf)

    c = sub.add_parser("compare", help="compare llama-server against reference scores")
    c.add_argument("--ref", default="ref.json")
    c.add_argument("--server", default="http://127.0.0.1:8080")
    c.add_argument("--max-abs", type=float, default=0.05, help="max allowed |score diff| (Q8_0 vs bf16)")
    c.add_argument("--min-spearman", type=float, default=0.98)
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=compare)

    args = p.parse_args()
    args.func(args)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
