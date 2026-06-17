"""
precompute.py — Parakh AI
Run ONCE before the timed ranking step.
Reads all 100K candidates, extracts structured features, saves to disk.
Also generates sentence-transformer embeddings for the semantic layer.

Usage:
    python precompute.py --candidates ./candidates.jsonl --out ./artifacts/

Runtime: ~8-15 minutes on CPU (embedding 100K profiles).
This is NOT subject to the 5-minute ranking constraint.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import compute_features, final_score


def build_candidate_text(c: dict) -> str:
    """
    Construct a rich text representation of the candidate for embedding.
    Prioritises career descriptions and skills over summary boilerplate.
    """
    parts = []

    # Title + headline
    parts.append(c["profile"]["current_title"])
    parts.append(c["profile"].get("headline", ""))

    # Career history: titles + descriptions (most signal-dense)
    for role in c["career_history"][:4]:  # top 4 roles
        parts.append(f"{role['title']} at {role['company']}")
        parts.append(role.get("description", "")[:400])

    # Skills (name + proficiency)
    skill_str = ", ".join(
        f"{s['name']} ({s['proficiency']})"
        for s in sorted(c["skills"], key=lambda x: {"expert":4,"advanced":3,"intermediate":2,"beginner":1}.get(x["proficiency"],0), reverse=True)[:15]
    )
    if skill_str:
        parts.append(skill_str)

    # Education
    for edu in c.get("education", [])[:2]:
        parts.append(f"{edu.get('degree','')} {edu.get('field_of_study','')} {edu.get('institution','')}")

    # Summary (last — often boilerplate)
    parts.append(c["profile"].get("summary", "")[:300])

    return " | ".join(p.strip() for p in parts if p.strip())


def main():
    parser = argparse.ArgumentParser(description="Parakh AI — Pre-computation step")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./artifacts",
                        help="Output directory for pre-computed artifacts")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding generation (faster, uses structured features only)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Parakh AI] Loading candidates from {args.candidates}")
    t0 = time.time()

    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    print(f"  Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # Step 1: Structured feature extraction
    # -----------------------------------------------------------------------
    print("\n[Step 1] Extracting structured features...")
    t1 = time.time()

    feature_rows = []
    iterator = tqdm(candidates, desc="Features") if HAS_TQDM else candidates
    for c in iterator:
        feat = compute_features(c)
        feat["composite_score"] = final_score(feat)
        feature_rows.append(feat)

    # Save as numpy-friendly arrays via a simple dict-of-arrays
    ids = np.array([r["candidate_id"] for r in feature_rows])
    scores = np.array([r["composite_score"] for r in feature_rows], dtype=np.float32)
    honeypots = np.array([r["is_honeypot"] for r in feature_rows], dtype=bool)

    np.save(out_dir / "candidate_ids.npy", ids)
    np.save(out_dir / "composite_scores.npy", scores)
    np.save(out_dir / "honeypot_flags.npy", honeypots)

    # Save full feature rows as jsonl for reasoning generation
    with open(out_dir / "features.jsonl", "w", encoding="utf-8") as f:
        for row in feature_rows:
            f.write(json.dumps(row) + "\n")

    print(f"  Structured features saved in {time.time()-t1:.1f}s")
    print(f"  Honeypots detected: {honeypots.sum()}")
    print(f"  Candidates with score > 0.50: {(scores > 0.50).sum()}")
    print(f"  Candidates with score > 0.30: {(scores > 0.30).sum()}")

    # -----------------------------------------------------------------------
    # Step 2: Embedding generation (semantic layer)
    # -----------------------------------------------------------------------
    if args.skip_embeddings:
        print("\n[Step 2] Skipping embeddings (--skip-embeddings flag set)")
        print("  Ranking will use structured features only.")
        return

    print("\n[Step 2] Generating embeddings (sentence-transformers)...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not installed. Run:")
        print("  pip install sentence-transformers")
        print("  Skipping embedding generation.")
        return

    t2 = time.time()
    # all-MiniLM-L6-v2: 384-dim, ~80MB, fast on CPU, strong for job-matching
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("  Building candidate texts...")
    texts = [build_candidate_text(c) for c in candidates]

    print(f"  Encoding {len(texts):,} candidates (batch_size=256)...")
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 norm → cosine sim = dot product
        convert_to_numpy=True,
    )  # shape: (100000, 384)

    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
    print(f"  Embeddings saved: shape={embeddings.shape}, {time.time()-t2:.1f}s")

    # -----------------------------------------------------------------------
    # Step 3: Build JD embedding
    # -----------------------------------------------------------------------
    print("\n[Step 3] Encoding job description...")
    JD_TEXT = """
    Senior AI Engineer Founding Team Redrob AI Series A AI-native talent intelligence platform.
    5 to 9 years experience. Production experience with embeddings-based retrieval systems
    sentence-transformers OpenAI embeddings BGE E5. Vector databases hybrid search infrastructure
    Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS. Strong Python.
    Evaluation frameworks ranking systems NDCG MRR MAP offline online correlation A/B testing.
    LLM fine-tuning LoRA QLoRA PEFT. Learning to rank XGBoost neural. HR tech recruiting marketplace.
    Shipped end-to-end ranking search recommendation system real users production scale.
    Product company not consulting research. Located Pune Noida Hyderabad Bangalore Mumbai Delhi.
    Hybrid work scrappy product engineering attitude. Applied ML not pure research.
    """
    jd_embedding = model.encode([JD_TEXT], normalize_embeddings=True, convert_to_numpy=True)
    np.save(out_dir / "jd_embedding.npy", jd_embedding.astype(np.float32))

    print(f"\n[Parakh AI] Pre-computation complete. Total time: {time.time()-t0:.1f}s")
    print(f"Artifacts saved to: {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        size = f.stat().st_size / (1024*1024)
        print(f"  {f.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
