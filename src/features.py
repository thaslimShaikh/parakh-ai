"""
features.py — Parakh AI
Structured feature extraction for every candidate against the JD.
This runs during pre-computation (no time limit) and produces a
feature matrix that the ranking step loads in milliseconds.
"""

from __future__ import annotations
import json
import math
from datetime import date, datetime
from typing import Any

# ---------------------------------------------------------------------------
# JD-derived constants (hard-coded from the released job description)
# ---------------------------------------------------------------------------

# Titles that directly match the role (score 1.0)
DIRECT_MATCH_TITLES = {
    "ml engineer", "machine learning engineer", "senior machine learning engineer",
    "ai engineer", "senior ai engineer", "applied scientist", "nlp engineer",
    "ai research engineer", "recommendation systems engineer", "mlops engineer",
    "research engineer", "applied ml engineer",
}

# Titles that are strong adjacent matches (score 0.75)
ADJACENT_TITLES = {
    "data scientist", "junior ml engineer", "software engineer",
    "full stack developer", "cloud engineer", "backend engineer",
    "data engineer", "analytics engineer", "platform engineer",
    "search engineer", "backend developer",
}

# Titles that are pure keyword-stuffers or irrelevant (score 0.0 — hard disqualifier)
IRRELEVANT_TITLES = {
    "hr manager", "accountant", "mechanical engineer", "civil engineer",
    "graphic designer", "marketing manager", "content writer",
    "sales executive", "operations manager", "customer support",
    "business analyst", "project manager", "frontend engineer",
    "mobile developer", ".net developer", "java developer",
}

# Consulting firms — JD explicitly says consulting-only is a soft disqualifier
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree", "lti",
    "ltimindtree", "ibm india", "atos", "dxc", "unisys",
}

# Core technical skills the JD explicitly requires (embeddings, retrieval, ranking, eval)
CORE_SKILLS = {
    "sentence transformers", "embeddings", "vector search", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "retrieval", "information retrieval", "semantic search", "hybrid search",
    "learning to rank", "ndcg", "mrr", "ranking", "recommendation systems",
    "rag", "llm", "llms", "fine-tuning llms", "lora", "qlora", "peft",
    "transformers", "hugging face transformers", "bert", "python",
    "pytorch", "tensorflow", "xgboost", "lightgbm",
}

# Preferred locations (Pune/Noida explicitly; Tier-1 Indian cities acceptable)
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "bangalore", "bengaluru",
    "mumbai", "delhi", "gurgaon", "gurugram", "chennai",
    "new delhi", "delhi ncr",
}

# Keywords in career descriptions that signal real retrieval/search/ranking work
RETRIEVAL_WORK_SIGNALS = [
    "retrieval", "ranking", "recommendation", "semantic search",
    "vector", "embedding", "search engine", "recsys", "rec sys",
    "information retrieval", "ranking system", "hybrid search",
    "candidate retrieval", "learning to rank",
]

# Product company indicators in descriptions (vs research / consulting)
PRODUCT_SIGNALS = [
    "shipped", "launched", "production", "real users", "a/b test",
    "a/b testing", "startup", "series a", "series b", "saas",
    "platform", "product company", "scale", "deployed to production",
]

# Research-only red flags (JD says pure research without deployment is a disqualifier)
RESEARCH_ONLY_SIGNALS = [
    "academic", "research lab", "phd", "university", "iit", "iim",
    "published", "arxiv", "paper", "journal", "thesis",
]

TODAY = date.today()


# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

def detect_honeypot(c: dict) -> bool:
    """
    Return True if the candidate has an impossible / fabricated profile.
    Checks:
      1. Claimed YOE vs sum of career history durations (>24 months gap = suspicious)
      2. 'Expert' proficiency skill with 0 duration_months
      3. Single role duration > claimed YOE (padding)
      4. Future end_date on a non-current role
    """
    yoe = c["profile"]["years_of_experience"]
    career = c["career_history"]
    skills = c["skills"]

    # Check 1: career months vs claimed YOE
    career_months = sum(r["duration_months"] for r in career)
    if career_months > (yoe * 12) + 30:   # 30-month tolerance
        return True

    # Check 2: expert skill with zero usage duration
    for s in skills:
        if s["proficiency"] == "expert" and s.get("duration_months", 1) == 0:
            return True

    # Check 3: single role longer than total claimed YOE
    for r in career:
        if r["duration_months"] > (yoe * 12) + 12:
            return True

    # Check 4: end_date in the future on a non-current role
    for r in career:
        if not r["is_current"] and r["end_date"]:
            try:
                end = datetime.strptime(r["end_date"], "%Y-%m-%d").date()
                if end > TODAY:
                    return True
            except ValueError:
                pass

    return False


# ---------------------------------------------------------------------------
# Individual feature extractors
# ---------------------------------------------------------------------------

def title_score(c: dict) -> float:
    """
    Title relevance — the single most decisive signal against keyword-stuffers.
    A Marketing Manager with 9 AI skills is still a Marketing Manager.
    """
    title = c["profile"]["current_title"].lower().strip()
    if any(t in title for t in DIRECT_MATCH_TITLES):
        return 1.0
    if any(t in title for t in ADJACENT_TITLES):
        return 0.65
    if any(t in title for t in IRRELEVANT_TITLES):
        return 0.0
    # Unknown title — give partial credit, fall back to career evidence
    return 0.35


def experience_score(c: dict) -> float:
    """
    JD wants 5-9 years. Peak at 6-8. Hard floor at 3, soft ceiling at 12.
    """
    yoe = c["profile"]["years_of_experience"]
    if 5 <= yoe <= 9:
        return 1.0
    if 4 <= yoe < 5:
        return 0.85
    if 9 < yoe <= 11:
        return 0.80
    if 3 <= yoe < 4:
        return 0.60
    if 11 < yoe <= 14:
        return 0.60
    if 2 <= yoe < 3:
        return 0.35
    return 0.10  # <2 or >14


def skill_match_score(c: dict) -> float:
    """
    Weighted skill match against JD core skills.
    Weight = proficiency multiplier * endorsement bonus * duration presence.
    This catches lazy keyword-stuffers: a skill listed with 'beginner' proficiency,
    0 endorsements, and 0 duration months gets almost no credit.
    """
    if not c["skills"]:
        return 0.0

    PROF_WEIGHT = {"expert": 1.0, "advanced": 0.80, "intermediate": 0.55, "beginner": 0.25}
    total_weight = 0.0
    matched_weight = 0.0

    for s in c["skills"]:
        name = s["name"].lower().strip()
        prof = PROF_WEIGHT.get(s["proficiency"], 0.3)
        endorse = min(s["endorsements"], 50) / 50.0   # cap at 50
        dur = s.get("duration_months", 0)
        dur_factor = min(dur, 36) / 36.0 if dur > 0 else 0.2  # 0.2 floor for 0-duration

        weight = prof * (0.7 + 0.2 * endorse + 0.1 * dur_factor)

        if name in CORE_SKILLS:
            matched_weight += weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    # Also add direct assessment score bonus
    assessments = c["redrob_signals"].get("skill_assessment_scores", {})
    assessed_core = [
        v / 100.0 for k, v in assessments.items()
        if k.lower() in CORE_SKILLS
    ]
    assessment_bonus = sum(assessed_core) / max(len(assessed_core), 1) * 0.15 if assessed_core else 0.0

    raw = matched_weight / (total_weight + 1e-9)
    return min(raw + assessment_bonus, 1.0)


def career_quality_score(c: dict) -> float:
    """
    Scores career trajectory for the 'product company, not pure consulting' signal.
    Rewards: product companies, AI/ML titles in history, search/recsys descriptions.
    Penalises: consulting-only background, pure research without deployment.
    """
    career = c["career_history"]
    if not career:
        return 0.0

    product_months = 0
    consulting_months = 0
    retrieval_evidence = 0
    product_evidence = 0
    total_months = sum(r["duration_months"] for r in career) or 1

    for r in career:
        company_lower = r["company"].lower()
        desc_lower = r["description"].lower()
        months = r["duration_months"]

        if any(cf in company_lower for cf in CONSULTING_FIRMS):
            consulting_months += months
        else:
            product_months += months

        retrieval_evidence += sum(1 for sig in RETRIEVAL_WORK_SIGNALS if sig in desc_lower)
        product_evidence += sum(1 for sig in PRODUCT_SIGNALS if sig in desc_lower)

    consulting_ratio = consulting_months / total_months
    product_ratio = product_months / total_months

    base = product_ratio * 0.6  # 60% weight on non-consulting tenure
    retrieval_bonus = min(retrieval_evidence / 5.0, 1.0) * 0.25
    product_bonus = min(product_evidence / 3.0, 1.0) * 0.15

    # Hard penalty for pure consulting background
    if consulting_ratio > 0.85:
        base *= 0.4

    return min(base + retrieval_bonus + product_bonus, 1.0)


def location_score(c: dict) -> float:
    """
    Pune/Noida preferred; Tier-1 Indian cities acceptable; willing_to_relocate saves others.
    """
    sig = c["redrob_signals"]
    location = c["profile"]["location"].lower()
    country = c["profile"]["country"].lower()

    if any(pl in location for pl in PREFERRED_LOCATIONS):
        return 1.0
    if country == "india" and sig.get("willing_to_relocate", False):
        return 0.80
    if country == "india":
        return 0.55
    if sig.get("willing_to_relocate", False):
        return 0.35
    return 0.10


def behavioral_modifier(c: dict) -> float:
    """
    The behavioral signal modifier (0.0 - 1.0) applied multiplicatively.
    A perfect-on-paper candidate who is effectively unreachable gets down-weighted.
    This is not a disqualifier on its own — it's a multiplier.
    """
    sig = c["redrob_signals"]

    # Recency: how recently were they active?
    last_active_str = sig.get("last_active_date", "2020-01-01")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (TODAY - last_active).days
        if days_inactive <= 14:
            recency = 1.0
        elif days_inactive <= 30:
            recency = 0.90
        elif days_inactive <= 60:
            recency = 0.75
        elif days_inactive <= 90:
            recency = 0.55
        elif days_inactive <= 180:
            recency = 0.35
        else:
            recency = 0.15
    except ValueError:
        recency = 0.30

    open_flag = 1.0 if sig.get("open_to_work_flag", False) else 0.55
    response_rate = sig.get("recruiter_response_rate", 0.5)
    interview_rate = sig.get("interview_completion_rate", 0.5)
    notice = sig.get("notice_period_days", 60)
    notice_score = 1.0 if notice <= 30 else (0.85 if notice <= 60 else (0.65 if notice <= 90 else 0.40))

    # GitHub activity: active public code = positive signal for an AI engineering role
    github = sig.get("github_activity_score", -1)
    github_score = (github / 100.0) * 0.8 + 0.2 if github >= 0 else 0.40

    # Profile completeness and verifications
    completeness = sig.get("profile_completeness_score", 50) / 100.0
    verified = (sig.get("verified_email", False) and sig.get("verified_phone", False))
    verify_bonus = 0.05 if verified else 0.0

    # Weighted combination
    modifier = (
        recency        * 0.30 +
        open_flag      * 0.20 +
        response_rate  * 0.18 +
        interview_rate * 0.12 +
        notice_score   * 0.08 +
        github_score   * 0.07 +
        completeness   * 0.05
    ) + verify_bonus

    return min(max(modifier, 0.05), 1.0)


def education_score(c: dict) -> float:
    """
    Mild signal. Tier-1 institution gives a small boost; no degree isn't penalised
    heavily since the JD explicitly says it cares about systems built, not pedigree.
    """
    edu = c.get("education", [])
    if not edu:
        return 0.40

    tier_scores = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.55, "tier_4": 0.35, "unknown": 0.45}
    best = max(tier_scores.get(e.get("tier", "unknown"), 0.45) for e in edu)

    # Relevant field bonus
    relevant_fields = {"computer science", "ai", "ml", "data science", "information technology",
                       "electronics", "electrical", "mathematics", "statistics"}
    field_bonus = 0.10 if any(
        any(rf in e.get("field_of_study", "").lower() for rf in relevant_fields)
        for e in edu
    ) else 0.0

    return min(best + field_bonus, 1.0)


# ---------------------------------------------------------------------------
# Final composite score
# ---------------------------------------------------------------------------

WEIGHTS = {
    "title":    0.28,   # Most decisive against keyword stuffers
    "career":   0.22,   # Product company, retrieval work evidence
    "skill":    0.20,   # Weighted skill match (not raw count)
    "exp":      0.12,   # YOE in the 5-9yr sweet spot
    "location": 0.08,   # Pune/Noida preference
    "edu":      0.05,   # Mild pedigree signal
    # behavioral modifier applied multiplicatively at the end
}

def compute_features(c: dict) -> dict:
    """Extract all features for one candidate. Returns a flat dict."""
    return {
        "candidate_id":    c["candidate_id"],
        "title_score":     title_score(c),
        "career_score":    career_quality_score(c),
        "skill_score":     skill_match_score(c),
        "exp_score":       experience_score(c),
        "location_score":  location_score(c),
        "edu_score":       education_score(c),
        "behavioral_mod":  behavioral_modifier(c),
        "is_honeypot":     detect_honeypot(c),
        # raw signals for reasoning generation
        "yoe":             c["profile"]["years_of_experience"],
        "title":           c["profile"]["current_title"],
        "location":        c["profile"]["location"],
        "country":         c["profile"]["country"],
        "open_to_work":    c["redrob_signals"]["open_to_work_flag"],
        "notice_days":     c["redrob_signals"]["notice_period_days"],
        "response_rate":   c["redrob_signals"]["recruiter_response_rate"],
        "last_active":     c["redrob_signals"]["last_active_date"],
        "github_score":    c["redrob_signals"]["github_activity_score"],
    }


def final_score(feat: dict) -> float:
    """
    Composite score = weighted_sum * behavioral_modifier.
    Honeypots are forced to near-zero.
    """
    if feat["is_honeypot"]:
        return 0.001

    # Hard zero for irrelevant titles (keyword stuffers)
    if feat["title_score"] == 0.0:
        return feat["behavioral_mod"] * 0.05   # tiny residual, won't reach top 100

    weighted = (
        feat["title_score"]    * WEIGHTS["title"] +
        feat["career_score"]   * WEIGHTS["career"] +
        feat["skill_score"]    * WEIGHTS["skill"] +
        feat["exp_score"]      * WEIGHTS["exp"] +
        feat["location_score"] * WEIGHTS["location"] +
        feat["edu_score"]      * WEIGHTS["edu"]
    )
    return round(weighted * feat["behavioral_mod"], 6)
