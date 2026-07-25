import time
import requests
import json
import os

API_RAG_BASE = os.getenv("RAG_BASE", "http://localhost:8000")

def run_chunking_benchmark():
    print("=" * 60)
    print("RAG Performance Benchmark — Chunking & Embedding Throughput")
    print("=" * 60)

    # Sample text (~100 KB text block)
    sample_paragraph = (
        "Wasserstein GAN (WGAN) introduces Earth Mover's Distance to overcome "
        "mode collapse and vanishing gradients during generative adversarial training. "
        "The critic network enforces a 1-Lipschitz continuity constraint via weight clipping "
        "or gradient penalty.\n\n"
    ) * 300  # ~100 KB

    # 1. Benchmark /chunk endpoint
    print(f"[Benchmark 1] Text Chunking Payload Size: {len(sample_paragraph) / 1024:.2f} KB")
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
            print(f"  ✓ Chunking completed in {t_chunk:.2f} ms ({len(chunks)} chunks produced)")
            print(f"  Throughput: {(len(sample_paragraph) / 1024) / (t_chunk / 1000):.2f} KB/s")
        else:
            print(f"  ✗ Chunking failed with status {res.status_code}: {res.text}")
    except Exception as e:
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
            print(f"  ✓ Embedding generated in {t_embed:.2f} ms (Dimension: {len(vec)})")
        else:
            print(f"  ✗ Embedding failed with status {res.status_code}: {res.text}")
    except Exception as e:
        print(f"  ✗ RAG service unreachable: {e}")

if __name__ == "__main__":
    run_chunking_benchmark()
