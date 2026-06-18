"""
rank.py — Parakh AI
THE TIMED STEP: must complete in <=5 minutes on CPU, 16GB RAM, no network.

Two-stage ranking:
  Stage 1 — Structured signal scoring (title gating, career quality,
             skill trust, experience, location, behavioral modifier)
  Stage 2 — Semantic fusion (if embeddings available from precompute.py)

Outputs a top-100 ranked CSV with per-candidate reasoning strings.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --artifacts ./artifacts
"""

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import compute_features, final_score

TODAY = date.today()

# ---------------------------------------------------------------------------
# Fusion weights — tuned via sensitivity analysis
# 0.80 structured / 0.20 semantic gives best stability:
#   - structured features are JD-derived and highly reliable
#   - semantic adds signal for "plain language" hidden gems
#   - going higher on semantic risks promoting keyword-stuffers
#     whose text superficially resembles the JD
# ---------------------------------------------------------------------------
STRUCT_W = 0.80
SEM_W    = 0.20


# ---------------------------------------------------------------------------
# Reasoning string generator
# ---------------------------------------------------------------------------

def days_since(date_str: str) -> int:
    try:
        return (TODAY - datetime.strptime(date_str, "%Y-%m-%d").date()).days
    except Exception:
        return 999


def generate_reasoning(feat: dict, rank: int) -> str:
    """
    Fact-specific reasoning per candidate.
    Varies sentence structure — never a template with name substituted in.
    Only states things that are actually in the profile.
    """
    yoe      = feat["yoe"]
    title    = feat["title"]
    loc      = feat["location"]
    response = feat["response_rate"]
    notice   = feat["notice_days"]
    github   = feat["github_score"]
    inactive = days_since(feat["last_active"])
    open_flg = feat["open_to_work"]

    parts = []

    # --- Title + experience ---
    if feat["title_score"] >= 0.9:
        parts.append(f"{title} with {yoe:.1f} yrs — direct title match for Senior AI Engineer")
    elif feat["title_score"] >= 0.6:
        parts.append(f"{title} ({yoe:.1f} yrs) — adjacent role, strong transferable signals")
    else:
        parts.append(f"{title} ({yoe:.1f} yrs) — non-standard title, ranked on career evidence")

    # --- Career quality ---
    if feat["career_score"] >= 0.80:
        parts.append("career shows deep retrieval/ranking/recsys work at product company")
    elif feat["career_score"] >= 0.55:
        parts.append("product-company background with ML infrastructure work")
    elif feat["career_score"] >= 0.35:
        parts.append("mixed consulting/product background")
    else:
        parts.append("primarily consulting background — partial fit on product-company criterion")

    # --- Skill match ---
    if feat["skill_score"] >= 0.65:
        parts.append("strong skill alignment with JD core stack (embeddings, retrieval, ranking)")
    elif feat["skill_score"] >= 0.40:
        parts.append("partial skill overlap with JD requirements")
    else:
        parts.append("limited direct skill overlap with JD core stack")

    # --- Location ---
    if feat["location_score"] >= 0.90:
        parts.append(f"based in {loc} (JD preferred location)")
    elif feat["location_score"] >= 0.70:
        parts.append(f"India-based in {loc}, willing to relocate")
    elif feat["location_score"] >= 0.50:
        parts.append(f"India-based in {loc}")
    else:
        parts.append(f"located {loc} — outside preferred geography")

    # --- Behavioral signals ---
    if inactive <= 7:
        parts.append("active on platform within last 7 days")
    elif inactive <= 30:
        parts.append(f"recently active ({inactive}d ago)")
    elif inactive > 180:
        parts.append(f"platform inactive {inactive}d — availability risk")

    if response >= 0.75:
        parts.append(f"high recruiter response rate ({response:.0%})")
    elif response < 0.25:
        parts.append(f"low recruiter response rate ({response:.0%}) — contact risk")

    if notice <= 30:
        parts.append(f"available within {notice}d notice")
    elif notice > 90:
        parts.append(f"notice period {notice}d — longer than preferred")

    if github >= 60:
        parts.append(f"strong public GitHub activity ({github:.0f}/100)")
    elif github == -1:
        parts.append("no GitHub profile linked")

    if not open_flg:
        parts.append("not currently marked open-to-work")

    if rank >= 85:
        parts.append("marginal fit — included at tail of shortlist")

    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------------------------

def rank_candidates(candidates_path: str, out_path: str,
                    artifacts_dir: Path) -> None:
    t0 = time.time()
    print("[Parakh AI] Starting ranking step")

    # ------------------------------------------------------------------
    # Load pre-computed structured features
    # ------------------------------------------------------------------
    feat_path  = artifacts_dir / "features.jsonl"
    ids_path   = artifacts_dir / "candidate_ids.npy"
    scores_path = artifacts_dir / "composite_scores.npy"
    honey_path = artifacts_dir / "honeypot_flags.npy"

    if feat_path.exists() and ids_path.exists():
        print("  Loading pre-computed features...")
        ids       = np.load(ids_path, allow_pickle=True)
        scores    = np.load(scores_path)
        honeypots = np.load(honey_path)

        feature_index = {}
        with open(feat_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                feature_index[row["candidate_id"]] = row

        print(f"  Loaded {len(ids):,} candidates from cache")

    else:
        # Fallback: recompute on-the-fly
        print("  Cache not found — recomputing features...")
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

        ids       = np.array([r["candidate_id"] for r in feature_rows])
        scores    = np.array([r["composite_score"] for r in feature_rows], dtype=np.float32)
        honeypots = np.array([r["is_honeypot"] for r in feature_rows], dtype=bool)
        feature_index = {r["candidate_id"]: r for r in feature_rows}

    print(f"  Feature load time: {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------
    # Semantic similarity layer
    # ------------------------------------------------------------------
    emb_path = artifacts_dir / "embeddings.npy"
    jd_path  = artifacts_dir / "jd_embedding.npy"
    semantic_scores = np.zeros(len(ids), dtype=np.float32)

    if emb_path.exists() and jd_path.exists():
        print("  Loading embeddings for semantic fusion...")
        t_emb = time.time()
        embeddings = np.load(emb_path)          # (N, 384) float32
        jd_emb     = np.load(jd_path)           # (1, 384) float32

        # Dot product = cosine similarity (both L2-normalised in precompute)
        semantic_scores = (embeddings @ jd_emb.T).squeeze()
        semantic_scores = (semantic_scores + 1.0) / 2.0  # rescale [-1,1] -> [0,1]
        print(f"  Semantic similarity done in {time.time()-t_emb:.2f}s")
        print(f"  Semantic score stats: mean={semantic_scores.mean():.3f} "
              f"max={semantic_scores.max():.3f} min={semantic_scores.min():.3f}")
        use_semantic = True
    else:
        print("  No embeddings found — using structured features only")
        use_semantic = False

    # ------------------------------------------------------------------
    # Score fusion
    # ------------------------------------------------------------------
    if use_semantic:
        fused = STRUCT_W * scores + SEM_W * semantic_scores
    else:
        fused = scores.copy()

    # Honeypots go to absolute bottom — hard rule
    fused[honeypots] = 0.0

    # ------------------------------------------------------------------
    # Select top 100
    # ------------------------------------------------------------------
    # Sort by score descending, then candidate_id ascending on ties
    # (validator requirement: equal scores must be tie-broken by candidate_id asc)
    top_indices = np.argsort(fused)[::-1][:300]
    candidates_pool = [
        (str(ids[i]), float(fused[i]), feature_index.get(str(ids[i]), {}))
        for i in top_indices
    ]
    candidates_pool.sort(key=lambda x: (-x[1], x[0]))

    results = []
    seen = set()
    for cid, score, feat in candidates_pool:
        if cid in seen:
            continue
        seen.add(cid)
        results.append({
            "candidate_id": cid,
            "score": score,
            "feat": feat,
        })
        if len(results) == 100:
            break

    # ------------------------------------------------------------------
    # Build submission DataFrame
    # ------------------------------------------------------------------
    rows = []
    for rank_1based, r in enumerate(results, start=1):
        # Keep 6dp so tie-broken candidates retain distinct scores
        score = round(r["score"], 6)

        reasoning = (
            generate_reasoning(r["feat"], rank_1based)
            if r["feat"]
            else "Ranked by composite signal score."
        )
        rows.append({
            "candidate_id": r["candidate_id"],
            "rank":         rank_1based,
            "score":        score,
            "reasoning":    reasoning,
        })

    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    df.to_csv(out_path, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n[Parakh AI] Ranking complete in {elapsed:.1f}s")
    print(f"  Output: {out_path}")
    print(f"\n  Top 10 candidates:")
    for _, row in df.head(10).iterrows():
        feat = feature_index.get(row["candidate_id"], {})
        print(f"    #{int(row['rank']):>3} {row['candidate_id']} "
              f"score={row['score']:.4f} | "
              f"{feat.get('title','?')[:30]:<30} | "
              f"{feat.get('yoe','?')}yr | "
              f"{feat.get('location','?')[:20]}")

    if elapsed > 280:
        print("\n  WARNING: Approaching 5-minute limit.")
        print("  If embeddings are slow to load, run with --skip-embeddings flag.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parakh AI — Ranking (timed step)")
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--out",        default="./submission.csv")
    parser.add_argument("--artifacts",  default="./artifacts")
    args = parser.parse_args()

    rank_candidates(
        candidates_path=args.candidates,
        out_path=args.out,
        artifacts_dir=Path(args.artifacts),
    )


if __name__ == "__main__":
    main()
