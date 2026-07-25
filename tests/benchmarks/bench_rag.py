import time
import requests
import json
import os
from datetime import datetime

API_RAG_BASE = os.getenv("RAG_BASE", "http://localhost:8000")
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

def run_chunking_benchmark():
    print("=" * 60)
    print("RAG Performance Benchmark — Chunking & Embedding Throughput")
    print("=" * 60)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Sample text (~100 KB text block)
    sample_paragraph = (
        "Wasserstein GAN (WGAN) introduces Earth Mover's Distance to overcome "
        "mode collapse and vanishing gradients during generative adversarial training. "
        "The critic network enforces a 1-Lipschitz continuity constraint via weight clipping "
        "or gradient penalty.\n\n"
    ) * 300  # ~100 KB

    results = {
        "timestamp": datetime.now().isoformat(),
        "payload_size_kb": len(sample_paragraph) / 1024,
        "chunking": {},
        "embedding": {},
    }

    # 1. Benchmark /chunk endpoint
    print(f"[Benchmark 1] Text Chunking Payload Size: {results['payload_size_kb']:.2f} KB")
    t0 = time.perf_counter()
    try:
        res = requests.post(f"{API_RAG_BASE}/chunk", json={
            "text": sample_paragraph,
            "chunk_size": 500,
            "overlap": 50
        }, timeout=10)
        t_chunk = (time.perf_counter() - t0) * 1000

        if res.status_code == 200:
            chunks = res.json().get("chunks", [])
            throughput = results['payload_size_kb'] / (t_chunk / 1000)
            results["chunking"] = {
                "success": True,
                "latency_ms": round(t_chunk, 2),
                "chunk_count": len(chunks),
                "throughput_kb_per_sec": round(throughput, 2)
            }
            print(f"  ✓ Chunking completed in {t_chunk:.2f} ms ({len(chunks)} chunks produced)")
            print(f"  Throughput: {throughput:.2f} KB/s")
        else:
            results["chunking"] = {"success": False, "error": res.text, "status_code": res.status_code}
            print(f"  ✗ Chunking failed with status {res.status_code}: {res.text}")
    except Exception as e:
        results["chunking"] = {"success": False, "error": str(e)}
        print(f"  ✗ RAG service unreachable: {e}")

    # 2. Benchmark /embed endpoint
    print(f"\n[Benchmark 2] Embedding Generation Latency")
    embed_query = "What is the Wasserstein distance in WGAN?"
    t0 = time.perf_counter()
    try:
        res = requests.post(f"{API_RAG_BASE}/embed", json={"text": embed_query}, timeout=10)
        t_embed = (time.perf_counter() - t0) * 1000

        if res.status_code == 200:
            vec = res.json().get("vector", [])
            results["embedding"] = {
                "success": True,
                "latency_ms": round(t_embed, 2),
                "dimension": len(vec)
            }
            print(f"  ✓ Embedding generated in {t_embed:.2f} ms (Dimension: {len(vec)})")
        else:
            results["embedding"] = {"success": False, "error": res.text, "status_code": res.status_code}
            print(f"  ✗ Embedding failed with status {res.status_code}: {res.text}")
    except Exception as e:
        results["embedding"] = {"success": False, "error": str(e)}
        print(f"  ✗ RAG service unreachable: {e}")

    # Export report JSON
    report_path = os.path.join(REPORTS_DIR, "rag_benchmark_summary.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Report Saved] Benchmark summary saved to: {report_path}")

if __name__ == "__main__":
    run_chunking_benchmark()
