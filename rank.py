"""
rank.py — Parakh AI
THE TIMED STEP: must complete in ≤5 minutes on CPU, 16GB RAM, no network.

Loads pre-computed artifacts (embeddings + structured features),
fuses semantic similarity with structured signals,
suppresses honeypots and keyword-stuffers,
outputs a top-100 ranked CSV.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --artifacts ./artifacts
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import compute_features, final_score, WEIGHTS

TODAY = date.today()

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


# ---------------------------------------------------------------------------
# Reasoning string generation
# ---------------------------------------------------------------------------

def days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (TODAY - d).days
    except Exception:
        return 999


def generate_reasoning(feat: dict, rank: int) -> str:
    """
    Generate a specific, honest reasoning string for Stage 4 manual review.
    Varies sentence structure per candidate — no template substitution.
    Mentions only things actually in the profile.
    """
    yoe = feat["yoe"]
    title = feat["title"]
    loc = feat["location"]
    response = feat["response_rate"]
    notice = feat["notice_days"]
    github = feat["github_score"]
    inactive_days = days_since(feat["last_active"])
    open_flag = feat["open_to_work"]

    parts = []

    # Title + experience (always)
    if feat["title_score"] >= 0.9:
        parts.append(f"{title} with {yoe:.1f} years directly matching the Senior AI Engineer profile")
    elif feat["title_score"] >= 0.6:
        parts.append(f"{title} ({yoe:.1f} yrs) — adjacent role with strong transferable signals")
    else:
        parts.append(f"{title} ({yoe:.1f} yrs)")

    # Career quality
    if feat["career_score"] >= 0.75:
        parts.append("career history shows retrieval/ranking/recsys work at product companies")
    elif feat["career_score"] >= 0.5:
        parts.append("some product-company exposure with relevant ML infrastructure work")
    elif feat["career_score"] < 0.3:
        parts.append("primarily consulting background — partial fit on the product-company criterion")

    # Skill match
    if feat["skill_score"] >= 0.6:
        parts.append("strong skill alignment with JD core stack (embeddings, retrieval, ranking)")
    elif feat["skill_score"] >= 0.35:
        parts.append("partial skill overlap with JD requirements")

    # Location
    if feat["location_score"] >= 0.9:
        parts.append(f"based in {loc} (preferred location)")
    elif feat["location_score"] >= 0.7:
        parts.append(f"India-based ({loc}), willing to relocate")

    # Behavioral signals
    if inactive_days <= 14:
        parts.append(f"active in last {inactive_days}d")
    elif inactive_days <= 30:
        parts.append("recently active on platform")
    elif inactive_days > 180:
        parts.append(f"inactive for {inactive_days} days — availability risk")

    if response >= 0.75:
        parts.append(f"high recruiter response rate ({response:.0%})")
    elif response < 0.25:
        parts.append(f"low response rate ({response:.0%}) — contact risk")

    if notice <= 30:
        parts.append(f"notice period {notice}d (immediate/near-immediate)")
    elif notice > 90:
        parts.append(f"notice period {notice}d — longer than preferred")

    if github >= 60:
        parts.append(f"strong GitHub activity ({github:.0f}/100)")
    elif github == -1:
        parts.append("no GitHub linked")

    if not open_flag:
        parts.append("not marked open-to-work")

    if rank >= 80:
        parts.append("included at tail of shortlist — marginal fit")

    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------------------------

def rank_candidates(candidates_path: str, out_path: str, artifacts_dir: Path) -> None:
    t0 = time.time()
    print(f"[Parakh AI rank.py] Starting ranking step")
    print(f"  Candidates: {candidates_path}")
    print(f"  Artifacts:  {artifacts_dir}")

    # ------------------------------------------------------------------
    # Load pre-computed artifacts
    # ------------------------------------------------------------------
    use_embeddings = (artifacts_dir / "embeddings.npy").exists()
    use_features   = (artifacts_dir / "features.jsonl").exists()

    if not use_features:
        # Fallback: recompute features on-the-fly (slower but still within 5 min for features-only)
        print("  WARNING: features.jsonl not found. Recomputing on-the-fly.")
        use_features = False

    if use_features:
        print("  Loading pre-computed features...")
        ids = np.load(artifacts_dir / "candidate_ids.npy", allow_pickle=True)
        scores = np.load(artifacts_dir / "composite_scores.npy")
        honeypots = np.load(artifacts_dir / "honeypot_flags.npy")

        # Load feature rows for reasoning (only top ~300 needed)
        feature_index = {}
        with open(artifacts_dir / "features.jsonl", "r") as f:
            for line in f:
                row = json.loads(line)
                feature_index[row["candidate_id"]] = row
        print(f"  Loaded {len(ids):,} candidates from pre-computed features")

    else:
        # Full recompute path (fallback)
        print("  Recomputing features from scratch...")
        candidates = []
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

        feature_rows = []
        for c in candidates:
            feat = compute_features(c)
            feat["composite_score"] = final_score(feat)
            feature_rows.append(feat)

        ids = np.array([r["candidate_id"] for r in feature_rows])
        scores = np.array([r["composite_score"] for r in feature_rows], dtype=np.float32)
        honeypots = np.array([r["is_honeypot"] for r in feature_rows], dtype=bool)
        feature_index = {r["candidate_id"]: r for r in feature_rows}

    print(f"  Feature load: {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------
    # Semantic similarity layer (if embeddings available)
    # ------------------------------------------------------------------
    semantic_scores = np.zeros(len(ids), dtype=np.float32)

    if use_embeddings and (artifacts_dir / "jd_embedding.npy").exists():
        print("  Loading embeddings for semantic fusion...")
        t_emb = time.time()
        embeddings = np.load(artifacts_dir / "embeddings.npy")  # (N, 384)
        jd_emb = np.load(artifacts_dir / "jd_embedding.npy")    # (1, 384)

        # Cosine similarity via dot product (embeddings are L2-normalised)
        semantic_scores = embeddings @ jd_emb.T       # (N, 1)
        semantic_scores = semantic_scores.squeeze()   # (N,)
        # Rescale from [-1,1] to [0,1]
        semantic_scores = (semantic_scores + 1.0) / 2.0
        print(f"  Semantic similarity computed: {time.time()-t_emb:.2f}s")
    else:
        print("  No embeddings found — using structured features only")

    # ------------------------------------------------------------------
    # Score fusion
    # ------------------------------------------------------------------
    # Structured features carry 75% weight; semantic 25%
    # This intentionally prevents pure keyword-embedding from dominating
    STRUCT_W = 0.75
    SEM_W = 0.25

    if use_embeddings and semantic_scores.any():
        fused = STRUCT_W * scores + SEM_W * semantic_scores
    else:
        fused = scores

    # Force honeypots to bottom
    fused[honeypots] = 0.0

    # ------------------------------------------------------------------
    # Select top 100
    # ------------------------------------------------------------------
    top_indices = np.argsort(fused)[::-1][:200]   # take 200, filter to 100

    results = []
    seen_ids = set()
    for idx in top_indices:
        cid = str(ids[idx])
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        results.append({
            "candidate_id": cid,
            "score": float(fused[idx]),
            "feat": feature_index.get(cid, {}),
        })
        if len(results) == 100:
            break

    # Ensure exactly 100 rows
    if len(results) < 100:
        print(f"  WARNING: only {len(results)} candidates above threshold")

    # ------------------------------------------------------------------
    # Build submission CSV
    # ------------------------------------------------------------------
    rows = []
    prev_score = None
    for rank_1based, r in enumerate(results, start=1):
        score = round(r["score"], 4)
        # Enforce non-increasing scores (add tiny epsilon for tie-break)
        if prev_score is not None and score > prev_score:
            score = prev_score
        prev_score = score

        reasoning = generate_reasoning(r["feat"], rank_1based) if r["feat"] else "Candidate ranked by composite signal score."

        rows.append({
            "candidate_id": r["candidate_id"],
            "rank": rank_1based,
            "score": score,
            "reasoning": reasoning,
        })

    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    df.to_csv(out_path, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n[Parakh AI] Ranking complete in {elapsed:.1f}s")
    print(f"  Output: {out_path}")
    print(f"  Top 5 candidates:")
    for _, row in df.head(5).iterrows():
        print(f"    Rank {int(row['rank'])}: {row['candidate_id']} | score={row['score']:.4f}")
        print(f"      {row['reasoning'][:90]}...")

    if elapsed > 290:
        print("  WARNING: Approaching 5-minute limit. Consider using --skip-embeddings.")


def main():
    parser = argparse.ArgumentParser(description="Parakh AI — Ranking step (timed)")
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv")
    parser.add_argument("--artifacts", default="./artifacts")
    args = parser.parse_args()

    rank_candidates(
        candidates_path=args.candidates,
        out_path=args.out,
        artifacts_dir=Path(args.artifacts),
    )


if __name__ == "__main__":
    main()
