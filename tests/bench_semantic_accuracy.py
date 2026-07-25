import os
import sys
import time
import statistics
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE     = os.getenv("API_BASE", "http://localhost:8080")
RAG_BASE     = os.getenv("RAG_BASE", "http://localhost:8000")
PDF_PATH     = Path(os.getenv("PDF_PATH", str(Path(__file__).parent.parent / "data" / "11 - WGAN.pdf")))
USERNAME     = "semcache_bench"
PASSWORD     = "semcache123"
CACHE_WORKERS = int(os.getenv("CACHE_WORKERS", "10"))

# Seconds to wait between LLM seed calls (quota protection - at least 5s)
LLM_INTER_QUERY_COOLDOWN = 10

# Seconds to wait after a cache miss (quota protection)
LLM_MISS_COOLDOWN = 12

# Results file
RESULTS_FILE = "tests/benchmark_results.json"

# Total request parameters for variants
# We will run each variant once to keep it in 'hundreds'
REPETITIONS_PER_VARIANT = 1
# We will use sequential execution to strictly follow the per-call cooldown rule
SEQUENTIAL_VARIANTS = True

QUERY_CLUSTERS = [
    {
        "topic": "Wasserstein Distance",
        "canonical": "What is the Wasserstein distance and why is it better than JS divergence for training GANs?",
        "variants": [
            "Explain the Wasserstein distance and its advantages over Jensen-Shannon divergence in GAN training",
            "Why does WGAN use Wasserstein distance instead of JS divergence?",
            "What makes Wasserstein distance superior to Jensen-Shannon divergence for generative models?",
            "How does the Wasserstein metric compare to JS divergence when optimizing GANs?",
            "Can you explain why Wasserstein distance is preferred over JS divergence in GAN training?",
            "Describe Wasserstein distance and explain why it is a better loss than JS divergence for GANs",
            "What is Earth Mover's distance and why is it better than JS divergence for training GANs?",
        ],
    },
    {
        "topic": "WGAN Critic Network",
        "canonical": "Describe the WGAN training algorithm and the role of the critic network",
        "variants": [
            "What is the role of the critic in WGAN and how does training work?",
            "How does the critic network in WGAN differ from a standard GAN discriminator?",
            "Explain the training procedure of WGAN and what the critic is doing",
            "What does the critic network do in Wasserstein GAN training?",
            "Describe how WGAN trains its critic and what makes it different from a discriminator",
            "Walk me through the WGAN algorithm and explain the critic's function",
            "In WGAN, what is the critic's role and how is the model trained step by step?",
        ],
    },
    {
        "topic": "GAN Evaluation Metrics",
        "canonical": "Please explain IS, FID, MMD, and how they are used for evaluation of GANs",
        "variants": [
            "What are Inception Score and FID and how are they used to evaluate GANs?",
            "Explain Frechet Inception Distance and Inception Score for evaluating generative models",
            "How do IS and FID measure the quality of GAN-generated images?",
            "Describe the evaluation metrics used for GANs including FID, IS, and MMD",
            "What metrics like FID and Inception Score tell us about GAN performance?",
            "How is GAN output quality measured using IS, FID, and Maximum Mean Discrepancy?",
            "Explain how FID, IS, and MMD are computed and used to benchmark GANs",
            "What do IS and FID measure when evaluating the outputs of a generative adversarial network?",
        ],
    },
    {
        "topic": "Mode Collapse",
        "canonical": "What is mode collapse in GANs and what techniques are used to prevent it?",
        "variants": [
            "Explain mode collapse in generative adversarial networks and how it can be avoided",
            "How does mode collapse happen in GANs and what methods exist to prevent it?",
            "What causes mode collapse in GAN training and what are the solutions?",
            "Describe the mode collapse problem in GANs and techniques to address it",
            "Why do GANs suffer from mode collapse and how can we fix it?",
            "What is mode collapse and what strategies mitigate it during GAN training?",
            "Can you explain mode collapse and the approaches taken to prevent it in GANs?",
        ],
    },
    {
        "topic": "Gradient Penalty",
        "canonical": "What is gradient penalty in WGAN-GP and why is it used instead of weight clipping?",
        "variants": [
            "Explain gradient penalty in WGAN-GP and its advantage over weight clipping",
            "Why does WGAN-GP use gradient penalty instead of clipping weights?",
            "What is the gradient penalty term in WGAN-GP and why replace weight clipping with it?",
            "How does gradient penalty work in WGAN-GP and what problem does it solve?",
            "Describe gradient penalty and explain why weight clipping was replaced in WGAN-GP",
            "What motivated the switch from weight clipping to gradient penalty in WGAN?",
            "Why is gradient penalty better than weight clipping for enforcing Lipschitz continuity in WGAN?",
        ],
    },
]

# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class HopTimings:
    """Per-request breakdown of network hops."""
    embed_ms: float = 0.0      # /embed call latency (ms)
    l1_cache_ms: float = 0.0   # Redis Stack Semantic Cache lookup (ms)
    total_ms: float = 0.0      # End-to-end Go backend latency (ms)
    cache_hit: bool = False
    success: bool = False
    status_code: int = 0
    error: Optional[str] = None


@dataclass
class QueryResult:
    question: str
    topic: str
    is_canonical: bool
    timings: HopTimings


# ── HTTP Helpers ───────────────────────────────────────────────────────────────

session = requests.Session()

def timed_post(url, **kwargs) -> tuple[requests.Response, float]:
    """Returns (response, elapsed_ms)."""
    t0 = time.perf_counter()
    resp = session.post(url, **kwargs)
    return resp, (time.perf_counter() - t0) * 1000


def measure_embed_latency(text: str) -> float:
    """Measure the Python RAG /embed round-trip in ms. Returns -1 on failure."""
    try:
        _, ms = timed_post(f"{RAG_BASE}/embed", json={"text": text}, timeout=10)
        return ms
    except Exception:
        return -1.0


# ── Steps ──────────────────────────────────────────────────────────────────────

def step_authenticate() -> dict:
    print_header("Step 1 — Authentication")
    for attempt, (endpoint, code) in enumerate([
        ("/api/v1/login", 200),
        ("/api/v1/register", 201),
    ]):
        resp, ms = timed_post(
            f"{API_BASE}{endpoint}",
            json={"username": USERNAME, "password": PASSWORD},
            timeout=10,
        )
        if resp.status_code == code:
            token = resp.json()["token"]
            verb = "Logged in" if attempt == 0 else "Registered"
            print(f"  ✓ {verb} as '{USERNAME}'  ({ms:.1f} ms)")
            return {"Authorization": f"Bearer {token}"}

    print(f"  ✗ Auth failed: {resp.status_code} — {resp.text}")
    sys.exit(1)


def step_ingest(headers: dict) -> str:
    print_header("Step 2 — Document Ingestion")
    if not PDF_PATH.exists():
        print(f"  ✗ PDF not found: {PDF_PATH}")
        sys.exit(1)

    with open(PDF_PATH, "rb") as f:
        resp, ms = timed_post(
            f"{API_BASE}/api/v1/ingest",
            headers=headers,
            files={"file": (PDF_PATH.name, f)},
            timeout=300,
        )

    if resp.status_code == 200:
        data = resp.json()
        doc_id = data["doc_id"]
        print(f"  ✓ '{PDF_PATH.name}' → {data['chunk_count']} chunks  ({ms:.1f} ms)")
        print(f"  doc_id: {doc_id}")
        return doc_id

    print(f"  ✗ Ingestion failed: {resp.status_code} — {resp.text}")
    sys.exit(1)


def step_seed_llm(headers: dict, doc_id: str) -> list[QueryResult]:
    """
    Fire each canonical query ONCE, sequentially, with a cooldown between them.
    This is the ONLY phase that touches the LLM APIs.
    Returns a list of QueryResult objects with hop timings.
    """
    print_header(
        f"Step 3 — LLM Seeding  [{len(QUERY_CLUSTERS)} canonical queries, sequential]"
    )
    print(f"  ⚠  Quota-safe: {LLM_INTER_QUERY_COOLDOWN}s cooldown between API calls\n")

    results: list[QueryResult] = []

    for i, cluster in enumerate(QUERY_CLUSTERS, 1):
        question = cluster["canonical"]
        short_q = question[:70]
        print(f"  [{i}/{len(QUERY_CLUSTERS)}] {short_q}...")

        # Measure /embed latency independently (not counted in Go backend time)
        embed_ms = measure_embed_latency(question)

        t0 = time.perf_counter()
        resp, total_ms = timed_post(
            f"{API_BASE}/api/v1/query",
            headers=headers,
            json={"question": question, "doc_id": doc_id, "top_k": 5},
            timeout=300,
        )
        success = resp.status_code == 200
        body = resp.json() if success else {}

        cache_hit = body.get("cache_hit", False)
        status_sym = "✓ (already cached)" if cache_hit else ("✓" if success else "✗")
        print(f"         {status_sym}  embed={embed_ms:.1f}ms  total={total_ms:.1f}ms")
        if not success:
            print(f"         Error {resp.status_code}: {resp.text[:120]}")

        results.append(QueryResult(
            question=question,
            topic=cluster["topic"],
            is_canonical=True,
            timings=HopTimings(
                embed_ms=embed_ms,
                total_ms=total_ms,
                cache_hit=cache_hit,
                success=success,
                status_code=resp.status_code,
            ),
        ))

        if i < len(QUERY_CLUSTERS):
            print(f"         💤 cooling down {LLM_INTER_QUERY_COOLDOWN}s...\n")
            time.sleep(LLM_INTER_QUERY_COOLDOWN)

    return results


def _fire_variant(headers, doc_id, question) -> HopTimings:
    """Single variant request — measures embed + end-to-end go-backend hop."""
    embed_ms = measure_embed_latency(question)

    resp, total_ms = timed_post(
        f"{API_BASE}/api/v1/query",
        headers=headers,
        json={"question": question, "doc_id": doc_id, "top_k": 5},
        timeout=90,
    )
    success = resp.status_code == 200
    body = resp.json() if success else {}
    cache_hit = body.get("cache_hit", False)

    return HopTimings(
        embed_ms=embed_ms,
        total_ms=total_ms,
        cache_hit=cache_hit,
        success=success,
        status_code=resp.status_code,
        error=None if success else f"HTTP {resp.status_code}",
    )


def step_variant_cache_test(headers: dict, doc_id: str) -> list[QueryResult]:
    """
    Fire a subset of semantic variants sequentially to honor quota safety.
    If a miss occurs, a long cooldown is triggered to be 'smart'.
    """
    all_variants = []
    for cluster in QUERY_CLUSTERS:
        for variant in cluster["variants"]:
            all_variants.append((cluster["topic"], variant))

    print_header(
        f"Step 4 — Semantic Variant Cache Test  [{len(all_variants)} unique queries, sequential]"
    )
    print(f"  ℹ  Sequential mode: honoring {LLM_MISS_COOLDOWN}s cooldown after any cache miss.")
    print(f"     Expected: ALL should ideally hit L1/L2 cache.\n")

    results: list[QueryResult] = []

    for i, (topic, question) in enumerate(all_variants, 1):
        print(f"  [{i}/{len(all_variants)}] {topic}: {question[:50]}...")
        
        timings = _fire_variant(headers, doc_id, question)
        
        hit_sym = "✓ HIT " if timings.cache_hit else "✗ MISS"
        print(f"         {hit_sym}  embed={timings.embed_ms:6.1f}ms  total={timings.total_ms:6.1f}ms")
        
        results.append(QueryResult(
            question=question,
            topic=topic,
            is_canonical=False,
            timings=timings,
        ))

        # Quota Safety: Only sleep if it was likely a miss that hit the LLM
        if not timings.cache_hit:
            print(f"         ⚠ Cache miss detection. Quota protection: {LLM_MISS_COOLDOWN}s cooldown...\n")
            time.sleep(LLM_MISS_COOLDOWN)
        else:
            # Small blink after a hit just to not hammer the Go backend
            time.sleep(0.1)

    return results


# ── Reporting ──────────────────────────────────────────────────────────────────

def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = max(0, int(len(s) * p / 100) - 1)
    return s[idx]


def print_latency_table(label: str, timings: list[HopTimings]):
    embed_ms = [t.embed_ms for t in timings if t.embed_ms > 0 and t.success]
    total_ms = [t.total_ms for t in timings if t.success]
    errors   = sum(1 for t in timings if not t.success)
    hits     = sum(1 for t in timings if t.cache_hit)

    print(f"\n  ┌─ {label}")
    print(f"  │  Requests : {len(timings)} total  |  {len(total_ms)} ok  |  {errors} errors")
    print(f"  │  Cache hit: {hits}/{len(timings)} ({100*hits/max(1,len(timings)):.1f}%)")
    print(f"  │")

    if embed_ms:
        print(f"  │  /embed latency (Python RAG hop)")
        print(f"  │    avg={statistics.mean(embed_ms):.1f}ms  "
              f"p50={percentile(embed_ms,50):.1f}ms  "
              f"p95={percentile(embed_ms,95):.1f}ms  "
              f"p99={percentile(embed_ms,99):.1f}ms  "
              f"min={min(embed_ms):.1f}ms  max={max(embed_ms):.1f}ms")
        print(f"  │")

    if total_ms:
        print(f"  │  end-to-end Go latency (all hops incl. cache/LLM)")
        print(f"  │    avg={statistics.mean(total_ms):.1f}ms  "
              f"p50={percentile(total_ms,50):.1f}ms  "
              f"p95={percentile(total_ms,95):.1f}ms  "
              f"p99={percentile(total_ms,99):.1f}ms  "
              f"min={min(total_ms):.1f}ms  max={max(total_ms):.1f}ms")

    print(f"  └{'─'*60}")


def print_per_topic_breakdown(results: list[QueryResult]):
    """Breakdown cache hit rate and latency per topic cluster."""
    from collections import defaultdict
    by_topic: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        by_topic[r.topic].append(r)

    print(f"\n  {'Topic':<30}  {'Variants':>8}  {'Hits':>6}  {'Hit%':>6}  {'avg total':>10}  {'p95 total':>10}")
    print(f"  {'─'*30}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*10}")
    for topic, cluster_results in sorted(by_topic.items()):
        hits = sum(1 for r in cluster_results if r.timings.cache_hit)
        n    = len(cluster_results)
        total_ms = [r.timings.total_ms for r in cluster_results if r.timings.success]
        avg  = statistics.mean(total_ms) if total_ms else 0.0
        p95  = percentile(total_ms, 95) if total_ms else 0.0
        sym  = "✓" if hits == n else ("⚠" if hits > 0 else "✗")
        print(f"  {sym} {topic:<28}  {n:>8}  {hits:>6}  {100*hits/max(1,n):>5.1f}%  {avg:>8.1f}ms  {p95:>8.1f}ms")


def print_summary(llm_results: list[QueryResult], variant_results: list[QueryResult]):
    print_header("═══  Benchmark Summary  ═══")

    # Phase 1: LLM (canonical)
    print("\n  ── Phase 1: External LLM Pipeline (canonical queries, sequential) ──")
    print_latency_table("Canonical / Cold Path (full LLM chain)", [r.timings for r in llm_results])

    # Phase 2: Semantic Cache (variants)
    print("\n  ── Phase 2: Semantic Cache (variant queries, concurrent) ──")
    variant_timings = [r.timings for r in variant_results]
    print_latency_table("Variant / Warm Cache Path (C++ SIMD + Redis)", variant_timings)

    # Per-topic breakdown
    print_header("Per-Topic Cache Hit Breakdown")
    print_per_topic_breakdown(variant_results)

    # Save results
    save_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "llm_results": [
            {
                "topic": r.topic,
                "question": r.question,
                "latency_ms": r.timings.total_ms,
                "embed_ms": r.timings.embed_ms,
                "success": r.timings.success
            } for r in llm_results
        ],
        "cache_results": [
            {
                "topic": r.topic,
                "question": r.question,
                "latency_ms": r.timings.total_ms,
                "embed_ms": r.timings.embed_ms,
                "cache_hit": r.timings.cache_hit,
                "success": r.timings.success
            } for r in variant_results
        ]
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  ✓ Results saved to {RESULTS_FILE}")

    # Overall pass/fail
    hits   = sum(1 for r in variant_results if r.timings.cache_hit)
    misses = len(variant_results) - hits
    errors = sum(1 for r in variant_results if not r.timings.success)
    all_canonical_ok = all(r.timings.success for r in llm_results)
    all_variants_hit = misses == 0 and errors == 0

    print_header("Final Verdict")
    print(f"  LLM seeds succeeded   : {'✓ ALL' if all_canonical_ok else '✗ PARTIAL'}")
    print(f"  Variant cache hits    : {hits}/{len(variant_results)} ({100*hits/max(1,len(variant_results)):.1f}%)")
    print(f"  Variant cache misses  : {misses}")
    print(f"  Variant errors        : {errors}")

    if all_canonical_ok:
        avg_llm_ms   = statistics.mean([r.timings.total_ms for r in llm_results if r.timings.success])
        avg_cache_ms = statistics.mean([r.timings.total_ms for r in variant_results if r.timings.success])
        speedup = avg_llm_ms / avg_cache_ms if avg_cache_ms > 0 else 0
        print(f"\n  Speedup (LLM → cache) : {speedup:.0f}×  "
              f"({avg_llm_ms:.0f}ms → {avg_cache_ms:.0f}ms avg)")

    verdict = "✓ PASS" if (all_canonical_ok and hits > 0) else "✗ FAIL"
    miss_note = ""
    if misses > 0:
        miss_note = f" ({misses} variants didn't hit semantic cache — check similarity threshold)"
    print(f"\n  {'─'*56}")
    print(f"  Overall: {verdict}{miss_note}")
    print(f"  {'─'*56}\n")


# ── Print Helpers ──────────────────────────────────────────────────────────────

def print_header(title: str):
    print(f"\n[{title}]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("--- CouncilAI Semantic Cache Benchmark ---")
    print(f"\n  API:     {API_BASE}")
    print(f"  RAG:     {RAG_BASE}")
    print(f"  Doc:     {PDF_PATH}")
    print(f"  Workers: {CACHE_WORKERS} (variant phase)")
    print(f"  Clusters: {len(QUERY_CLUSTERS)} canonical queries")
    total_variants = sum(len(c["variants"]) for c in QUERY_CLUSTERS)
    print(f"  Variants: {total_variants} semantic rephrases")
    print(f"  LLM cooldown: {LLM_INTER_QUERY_COOLDOWN}s between queries (quota-safe)\n")

    headers = step_authenticate()
    doc_id  = step_ingest(headers)

    # Phase 1 — sequential LLM seeding (quota-safe)
    llm_results = step_seed_llm(headers, doc_id)

    # Phase 2 — concurrent variant cache test (no LLM calls)
    variant_results = step_variant_cache_test(headers, doc_id)

    # Report
    print_summary(llm_results, variant_results)


if __name__ == "__main__":
    main()
