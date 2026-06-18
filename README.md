# Parakh AI — Intelligent Candidate Discovery
### India Runs Hackathon · Track 1: Data & AI Challenge
**Team:** Parakh AI &nbsp;|&nbsp; **Participant:** Shaik Thaslim &nbsp;|&nbsp; **Institution:** Amrita Vishwa Vidyapeetham

> *"Parakh" (परख) — Hindi for discernment, judgement, the ability to see true worth.*

---

## What This Solves

Recruiters lose good candidates not because the talent isn't there — but because keyword filters can't see what actually matters. A Marketing Manager who lists "PyTorch" and "LLMs" in their skills section scores identically to a Senior ML Engineer who built a production recsys at a product startup. That's the failure this system is designed to fix.

Parakh AI ranks candidates **the way a great recruiter would** — by understanding who genuinely fits the role, not who wrote the best-optimised profile.

---

## Architecture

The system is a **two-stage pipeline** with strict separation between offline pre-computation and the timed online ranking step.

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Pre-computation (runs once, no time limit)           │
│                                                                  │
│  candidates.jsonl ──► Feature Extraction ──► features.jsonl    │
│                   ──► SentenceTransformer  ──► embeddings.npy   │
│                       (all-MiniLM-L6-v2)      jd_embedding.npy  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Ranking (timed: ≤5 min, CPU-only, no network)        │
│                                                                  │
│  Load cached features  ──► Structured score  ─┐                 │
│  Load embeddings       ──► Semantic score    ──┼► Fused score   │
│                                                │                 │
│  Fused = 0.80 × struct + 0.20 × semantic      │                 │
│  Honeypots → score = 0.0 (hard rule)          │                 │
│                                     └──► Top-100 CSV + reasons  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

The 5-minute CPU-only constraint makes GPU inference and live LLM calls impossible during ranking. The insight is that **the expensive work (embedding 100K profiles) can be pre-computed once** and cached — leaving the actual ranking step as a fast matrix multiply + feature lookup that completes in under 30 seconds.

---

## Scoring Logic

### Structured Score (80% weight)

Five signals, each derived directly from what the JD actually says it cares about:

| Signal | Weight | What it measures | Why it matters |
|--------|--------|-----------------|----------------|
| **Title Gate** | 28% | Is current title in the ML/AI engineering family? | A Marketing Manager with 9 AI skills is still a Marketing Manager. This is the primary anti-keyword-stuffer defence. |
| **Career Quality** | 22% | Product company tenure + retrieval/recsys work in descriptions | JD explicitly disqualifies consulting-only and pure-research backgrounds |
| **Skill Trust** | 20% | Weighted by proficiency × endorsements × duration | A skill listed as "expert" with 0 months duration and 0 endorsements gets 20% of the credit of a verified expert skill |
| **Experience** | 12% | Years in the 5–9yr JD sweet spot | Peak score at 6–8 years; graceful degradation outside range |
| **Location** | 8% | Pune/Noida preference; India + willing-to-relocate acceptable | JD states preference explicitly |
| **Education** | 5% | Tier 1/2 institution + relevant field | Mild signal; JD cares more about systems shipped |

**Behavioral Modifier (multiplicative):** Applied on top of the weighted sum, not added to it. A great-on-paper candidate who is platform-inactive for 6 months, has a 15% recruiter response rate, and isn't open-to-work gets their structural score multiplied down — not zeroed, but realistically deprioritised.

Behavioral signals used: recency of last platform activity, open-to-work flag, recruiter response rate, interview completion rate, notice period, GitHub activity score, profile completeness, email/phone verification.

### Semantic Score (20% weight)

Cosine similarity between the candidate's text representation (title + career descriptions + skills) and the JD embedding, using `all-MiniLM-L6-v2`. This catches "plain-language" candidates whose career descriptions demonstrate deep retrieval/search expertise without necessarily using JD buzzwords like "RAG" or "Pinecone."

The 80/20 split was chosen deliberately: heavier semantic weighting risks promoting keyword-stuffers whose profiles superficially resemble the JD text. The structured layer already handles semantic fit via the career quality signal.

### Honeypot Detection

Hard rule — honeypot-flagged candidates receive a score of 0.0 and cannot appear in the top 100. Detection checks:
- Claimed YOE vs sum of career history durations (>30 month gap)
- "Expert" proficiency skill with 0 duration months
- Single role duration exceeding total claimed YOE
- Future end_date on a non-current role

43 honeypots detected in the 100K dataset.

---

## Repository Structure

```
parakh-ai/
├── precompute.py          # Stage 1: feature extraction + embedding generation
├── rank.py                # Stage 2: timed ranking step (≤5 min)
├── validate_submission.py # Official submission validator
├── requirements.txt
├── src/
│   └── features.py        # All feature extraction logic
└── artifacts/             # Pre-computed outputs (generated by precompute.py)
    ├── .gitkeep
    ├── candidate_ids.npy       # not tracked in git (large)
    ├── composite_scores.npy    # not tracked in git
    ├── embeddings.npy          # not tracked in git (~150MB)
    ├── jd_embedding.npy        # not tracked in git
    ├── honeypot_flags.npy      # not tracked in git
    └── features.jsonl          # not tracked in git (~45MB)
```

---

## Setup & Reproduction

### Requirements
- Python 3.9+
- CPU only (no GPU required)
- ~4GB RAM for ranking step; ~8GB for precompute with embeddings

```bash
git clone https://github.com/thaslimShaikh/parakh-ai.git
cd parakh-ai
pip install -r requirements.txt
```

### Step 1 — Pre-compute (run once, ~15 min on CPU)
```bash
python precompute.py \
    --candidates path/to/candidates.jsonl \
    --out ./artifacts
```

To skip embedding generation (faster, structured features only):
```bash
python precompute.py \
    --candidates path/to/candidates.jsonl \
    --out ./artifacts \
    --skip-embeddings
```

### Step 2 — Rank (timed step, runs in ~20-30 sec with embeddings)
```bash
python rank.py \
    --candidates path/to/candidates.jsonl \
    --out ./submission.csv \
    --artifacts ./artifacts
```

### Step 3 — Validate
```bash
python validate_submission.py submission.csv
# Expected output: Submission is valid.
```

---

## Key Design Decisions

**Why not a pure LLM ranker?**
The 5-minute CPU-only constraint makes live LLM inference impractical at 100K scale. More importantly, LLMs without structured grounding tend to reward well-written profiles over actually-qualified candidates — exactly the problem we're trying to solve.

**Why sentence-transformers over OpenAI embeddings?**
CPU-friendly, no network dependency during the timed step, and `all-MiniLM-L6-v2` (80MB) is a strong model for job-matching tasks. OpenAI embeddings require API calls, violating the no-network constraint on ranking.

**Why multiplicative behavioral modifier instead of additive?**
An additive signal can rescue a bad structural fit with good behavioral signals. Multiplication means: if you're not a fit on paper, being active on the platform doesn't promote you. If you are a great fit but unreachable, you get realistically downweighted — not eliminated.

**Why title gating at 28% weight?**
The JD literally describes the keyword-stuffer problem. Title is the single most reliable anti-gaming signal: it's set by the employer at hire time and is very hard to fake without the entire career history being fraudulent (which the honeypot detection catches separately).

---

## Results Summary

| Metric | Value |
|--------|-------|
| Total candidates evaluated | 100,000 |
| Honeypots detected & suppressed | 43 |
| Ranking step runtime | ~20 sec (with embeddings) |
| Ranking step runtime | ~14 sec (structured only) |
| Validator result | ✅ Submission is valid |
| Top candidate score | 0.8192 |
| Score range (top 100) | 0.58 – 0.82 |

---

## About

Built for the **India Runs Hackathon** by Redrob AI — a challenge to build the next generation of intelligent talent discovery systems for India.

**Shaik Thaslim** · Final Year B.Tech, AI & Engineering · Amrita Vishwa Vidyapeetham  
[LinkedIn](https://linkedin.com/in/shaik-thaslim-33081030a) · [GitHub](https://github.com/thaslimShaikh)
