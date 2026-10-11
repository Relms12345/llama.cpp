#!/usr/bin/env python3
"""Compare jina-reranker-v3.5 scores: Hugging Face reference vs llama-server /v1/rerank.

Regression test (no torch needed; uses the committed scripts/jina-rerank-ref.json):
    python scripts/compare-jina-rerank.py run --model jina-reranker-v3.5-Q8_0.gguf
This starts build/bin/llama-server once per entry in CONFIGS ("reference": -np 1 -c 131072,
"nix": the router preset in scripts/jina-rerank-nix.ini) and checks every case against the
reference. Use --config to pick one, --server-bin for another build, --extra for e.g. -ngl 99.

Refresh the reference after changing the cases (needs torch + transformers):
    python scripts/compare-jina-rerank.py dump-hf --model jinaai/jina-reranker-v3.5 --out scripts/jina-rerank-ref.json

Check a server that is already running:
    python scripts/compare-jina-rerank.py compare --server http://127.0.0.1:8080 [--model NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
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

    # about 24k tokens: one block at -c 131072 (like the reference), several blocks at small -c
    topics = ["coral reefs", "railway history", "quantum computing", "olive oil", "glaciers", "chess openings"]
    docs = []
    for i in range(60):
        t = topics[i % len(topics)]
        docs.append(f"Report {i} on {t}. " + f"This paragraph describes {t} and related facts in some depth. " * 30)
    cases.append({"name": "split_small_ctx", "query": "how do glaciers move?", "documents": docs})
    return cases


def select_cases(cases: list[dict], names: str | None) -> list[dict]:
    if not names:
        return cases
    wanted = names.split(",")
    return [c for c in cases if c["name"] in wanted]


def case_hash(case: dict) -> str:
    return hashlib.sha256(json.dumps([case["query"], case["documents"]]).encode()).hexdigest()


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
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype)
    model.to(args.device).eval()
    if not hasattr(model, "rerank"):
        sys.exit("model has no rerank() method; check the model card for the reference API")

    # only scores and a hash are stored; the cases are rebuilt from build_cases()
    out = {"model": args.model, "dtype": args.dtype, "cases": []}
    for case in select_cases(build_cases(), args.cases):
        with torch.no_grad():
            results = model.rerank(case["query"], case["documents"], top_n=len(case["documents"]))
        scores = normalize_results(results, len(case["documents"]))
        out["cases"].append({"name": case["name"], "sha256": case_hash(case), "ref_scores": scores})
        print(f"{case['name']}: {len(case['documents'])} docs")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


def load_refs(paths: list[str], names: str | None) -> tuple[dict, list[dict]]:
    """Merge dump-hf outputs and attach the query/documents of each case."""
    meta: dict = {}
    by_name = {c["name"]: c for c in build_cases()}
    cases = []
    for path in paths:
        with open(path) as f:
            r = json.load(f)
        meta.update({k: v for k, v in r.items() if k != "cases"})
        for c in r["cases"]:
            if "documents" not in c:
                src = by_name.get(c["name"])
                if src is None or case_hash(src) != c["sha256"]:
                    sys.exit(f"{path}: case {c['name']!r} differs from build_cases(); re-run dump-hf")
                c = dict(src, **c)
            cases.append(c)
    return meta, select_cases(cases, names)


def llama_scores(server: str, query: str, docs: list[str], model: str | None = None) -> list[float]:
    req_body = {"query": query, "documents": docs, "top_n": len(docs)}
    if model:
        req_body["model"] = model
    req = urllib.request.Request(server.rstrip("/") + "/v1/rerank", data=json.dumps(req_body).encode(),
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


def check(diffs: list[float], sp: float, n: int, top1: bool, top5: float, lim: dict) -> bool:
    return (max(diffs) <= lim["max_abs"] and (sp >= lim["min_spearman"] or n < 4)
            and (top1 or not lim.get("require_top1", True)) and top5 >= lim.get("min_top5", 0.0))


def compare_cases(cases: list[dict], server: str, model: str | None, max_abs: float, min_spearman: float,
                  min_top5: float, verbose: bool = False, known: dict | None = None) -> bool:
    """Score every case on the server and check it against the reference; returns True if all pass.
    known maps case names to looser limits for deviations the config is expected to have;
    such a case reports KNOWN when it misses the normal limits but stays within the loose ones."""
    strict = {"max_abs": max_abs, "min_spearman": min_spearman, "min_top5": min_top5}
    known = known or {}
    all_ok = True
    print(f"{'case':<16}{'docs':>5}{'max|d|':>9}{'mean|d|':>9}{'pearson':>9}{'spearman':>9}{'top1':>6}{'top5':>6}")
    for case in cases:
        a = case["ref_scores"]
        b = llama_scores(server, case["query"], case["documents"], model)
        diffs = [abs(x - y) for x, y in zip(a, b)]
        sp = pearson(ranks(a), ranks(b))
        top1 = topk(a, 1) == topk(b, 1)
        k = min(5, len(a))
        top5 = len(set(topk(a, k)) & set(topk(b, k))) / k
        ok = check(diffs, sp, len(a), top1, top5, strict)
        status = "OK"
        if not ok and case["name"] in known:
            ok = check(diffs, sp, len(a), top1, top5, known[case["name"]])
            status = "KNOWN" if ok else "FAIL"
        elif not ok:
            status = "FAIL"
        all_ok &= ok
        print(f"{case['name']:<16}{len(a):>5}{max(diffs):>9.4f}{sum(diffs) / len(diffs):>9.4f}"
              f"{pearson(a, b):>9.4f}{sp:>9.4f}{str(top1):>6}{top5:>6.2f}  {status}")
        if verbose or not ok:
            for i in topk(a, min(10, len(a))):
                print(f"    doc {i:>3}: ref {a[i]:+.4f}  llama {b[i]:+.4f}")
    return all_ok


def compare(args) -> None:
    meta, cases = load_refs(args.ref, args.cases)
    print(f"reference: {meta.get('model')} ({meta.get('dtype')})\n")
    ok = compare_cases(cases, args.server, args.model, args.max_abs, args.min_spearman, args.min_top5, args.verbose)
    print("\nPASS" if ok else "\nFAIL (see cases above)")
    sys.exit(0 if ok else 1)


# Server configurations checked by `run`. Tolerances are against the HF fp32 reference
# for a Q8_0 GGUF. "known" lists cases a config cannot match because its blocks are
# smaller than the reference's; they get loose bounds (measured on CPU plus a margin)
# so they still catch regressions.
NIX_SECTION = "hf.co/jinaai/jina-reranker-v3.5-GGUF:Q8_0"
CONFIGS = {
    # the reference block size: one slot with the full 131072-token context
    "reference": {
        "args": ["-m", "{model}", "--embedding", "--pooling", "none", "--attention", "causal",
                 "-np", "1", "-c", "131072"],
        "max_abs": 0.01, "min_spearman": 0.99, "min_top5": 1.0,
    },
    # router mode with the preset from relms/flake systems/llama.nix: 2 slots of 8192 tokens,
    # q8_0 KV cache, flash attention
    "nix": {
        "preset": "jina-rerank-nix.ini",
        "router_model": NIX_SECTION,
        "args": ["--models-preset", "{preset}", "--models-max", "3", "--models-autoload"],
        "max_abs": 0.01, "min_spearman": 0.9, "min_top5": 1.0,
        "known": {
            # 125-doc blocks do not fit in 8192 tokens; measured 0.0083 / 0.998 / 0.8
            "many_docs_300":   {"max_abs": 0.02, "min_spearman": 0.99, "min_top5": 0.8},
            # the 9k-token document is cut to fit an 8192-token block; measured 0.091
            "truncation":      {"max_abs": 0.12, "min_spearman": 0.98, "min_top5": 1.0},
            # 3 small blocks instead of 1; measured 0.105 / 0.71 / 0.8, top-1 wrong
            "split_small_ctx": {"max_abs": 0.15, "min_spearman": 0.6, "min_top5": 0.6, "require_top1": False},
        },
    },
    # the nix preset with the rerank model given one 131072-token slot (recommended)
    "nix-131k": {
        "preset": "jina-rerank-nix.ini",
        "preset_overrides": {NIX_SECTION: {"parallel": "1", "kv-unified-per-slot": "131072"}},
        "router_model": NIX_SECTION,
        "args": ["--models-preset", "{preset}", "--models-max", "3", "--models-autoload"],
        "max_abs": 0.01, "min_spearman": 0.9, "min_top5": 1.0,
    },
}


def apply_overrides(ini: str, overrides: dict) -> str:
    """Set keys in INI sections (string level; enough for the presets used here)."""
    lines = ini.splitlines()
    for section, kv in overrides.items():
        start = lines.index(f"[{section}]")
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("[")), len(lines))
        body = [ln for ln in lines[start + 1:end] if ln.split("=", 1)[0].strip() not in kv]
        body += [f"{k}={v}" for k, v in kv.items()]
        lines = lines[:start + 1] + body + lines[end:]
    return "\n".join(lines) + "\n"



def wait_healthy(url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server did not become healthy")


def run(args) -> None:
    meta, cases = load_refs(args.ref, args.cases)
    print(f"reference: {meta.get('model')} ({meta.get('dtype')})")
    names = args.config or list(CONFIGS)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}
    for name in names:
        cfg = CONFIGS[name]
        with tempfile.TemporaryDirectory() as tmp:
            subst = {"model": os.path.abspath(args.model)}
            if "preset" in cfg:
                with open(os.path.join(script_dir, cfg["preset"])) as f:
                    ini = apply_overrides(f.read().replace("{model}", subst["model"]), cfg.get("preset_overrides", {}))
                subst["preset"] = os.path.join(tmp, "preset.ini")
                with open(subst["preset"], "w") as f:
                    f.write(ini)
            cmd = [args.server_bin] + [a.format(**subst) for a in cfg["args"]]
            cmd += ["--host", "127.0.0.1", "--port", str(args.port)] + args.extra
            log_path = os.path.join(args.log_dir or tmp, f"llama-server-{name}.log")
            print(f"\n=== {name}: {' '.join(cmd)}")
            with open(log_path, "w") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                try:
                    url = f"http://127.0.0.1:{args.port}"
                    wait_healthy(url, proc, args.startup_timeout)
                    ok = compare_cases(cases, url, cfg.get("router_model"), cfg["max_abs"], cfg["min_spearman"],
                                       cfg["min_top5"], args.verbose, cfg.get("known"))
                except Exception as e:
                    print(f"error: {e}")
                    ok = False
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            if not ok:
                with open(log_path) as f:
                    tail = f.readlines()[-20:]
                print(f"--- last lines of {log_path}:\n" + "".join(tail))
        results[name] = ok
    print("\n" + "\n".join(f"{n:<12} {'PASS' if ok else 'FAIL'}" for n, ok in results.items()))
    sys.exit(0 if all(results.values()) else 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    default_ref = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "jina-rerank-ref.json")]

    d = sub.add_parser("dump-hf", help="compute reference scores with transformers")
    d.add_argument("--model", default="jinaai/jina-reranker-v3.5")
    d.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    d.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    d.add_argument("--out", default="ref.json")
    d.add_argument("--cases", help="comma-separated case names (default: all)")
    d.set_defaults(func=dump_hf)

    c = sub.add_parser("compare", help="compare a running llama-server against reference scores")
    c.add_argument("--ref", nargs="+", default=default_ref, help="one or more dump-hf outputs")
    c.add_argument("--cases", help="comma-separated case names (default: all)")
    c.add_argument("--server", default="http://127.0.0.1:8080")
    c.add_argument("--model", help="model name to send (router mode)")
    c.add_argument("--max-abs", type=float, default=0.01, help="max allowed |score diff| (Q8_0 vs fp32 is about 0.006)")
    c.add_argument("--min-spearman", type=float, default=0.98)
    c.add_argument("--min-top5", type=float, default=0.0, help="min top-5 overlap with the reference")
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=compare)

    r = sub.add_parser("run", help="start llama-server for each config in CONFIGS and compare (regression test)")
    r.add_argument("--model", required=True, help="GGUF file")
    r.add_argument("--server-bin", default="build/bin/llama-server")
    r.add_argument("--config", action="append", choices=list(CONFIGS), help="config to run (default: all)")
    r.add_argument("--ref", nargs="+", default=default_ref, help="one or more dump-hf outputs")
    r.add_argument("--cases", help="comma-separated case names (default: all)")
    r.add_argument("--port", type=int, default=8097)
    r.add_argument("--startup-timeout", type=float, default=600)
    r.add_argument("--log-dir", help="keep server logs here")
    r.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra llama-server args (must be last)")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=run)

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
