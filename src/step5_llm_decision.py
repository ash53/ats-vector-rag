"""
STEP 5 — LLM Decision Engine

Given a query CV and Job Description, this script:
1. Runs the full retrieval pipeline (Steps 3–4) inline
2. Formats the structured evidence chunks into a prompt
3. Calls Llama 3.2 via Ollama (localhost:11434)
4. Parses and returns a structured hiring decision

Output contract:
{
    "decision":      "select" | "reject",
    "confidence":    "high" | "medium" | "low",
    "key_strengths": ["..."],
    "key_gaps":      ["..."],
    "reasoning":     "..."
}
"""

# -------------------------------
# macOS / PyTorch stability fixes
# -------------------------------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -------------------------------
# Imports
# -------------------------------
import json
import requests
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# -------------------------------
# File paths
# -------------------------------
INDEX_PATH = Path("data/processed/faiss_index.bin")
META_PATH  = Path("data/processed/faiss_metadata.json")
CASES_PATH = Path("data/processed/cases_stage2.jsonl")

# -------------------------------
# LLM config
# -------------------------------
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "llama3.1:8b"
TIMEOUT_S   = 180

# -------------------------------
# Load index, metadata, cases
# -------------------------------
# Cached at module level: the index (62 MB), the case file (38 MB) and the
# embedding model are identical for every query, so loading them per call made
# a batch evaluation spend most of its time on disk I/O. Loaded once, lazily.
_ARTIFACTS = None
_MODEL = None


def _load_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS

    assert INDEX_PATH.exists(), "FAISS index not found. Run step3 first."
    assert META_PATH.exists(),  "FAISS metadata not found. Run step3 first."
    assert CASES_PATH.exists(), "cases_stage2.jsonl not found. Run step2 first."

    idx = faiss.read_index(str(INDEX_PATH))

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cases = {}
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cases[c["case_id"]] = c

    _ARTIFACTS = (idx, meta, cases)
    return _ARTIFACTS


def _load_model():
    global _MODEL
    if _MODEL is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _MODEL = SentenceTransformer("all-mpnet-base-v2", device=device)
    return _MODEL

# -------------------------------
# Embedding helpers (mirrors step3/4)
# -------------------------------
def _build_case_text(role, cv_text, jd_text):
    return f"""
    Role: {role}
    Resume: {cv_text}
    Job Description: {jd_text}
    """

def _build_skill_text(skills_cv, skills_jd):
    missing = sorted(set(skills_jd) - set(skills_cv))
    return f"""
    Candidate skills: {", ".join(skills_cv)}
    Job required skills: {", ".join(skills_jd)}
    Missing skills: {", ".join(missing)}
    """

def embed_query(model, role, cv_text, jd_text, skills_cv, skills_jd):
    case_emb  = model.encode([_build_case_text(role, cv_text, jd_text)],  normalize_embeddings=True)
    skill_emb = model.encode([_build_skill_text(skills_cv, skills_jd)], normalize_embeddings=True)
    return np.hstack([case_emb, skill_emb]).astype("float32")

# -------------------------------
# Retrieval (mirrors step4)
# -------------------------------
def retrieve_similar_cases(index, metadata, cases, query_vector, k=20, exclude_case_id=None):
    # Fetch extra candidates so we still get k results after excluding one
    fetch_k = k + 1 if exclude_case_id else k
    scores, indices = index.search(query_vector, fetch_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        meta = metadata[idx]
        if exclude_case_id and meta["case_id"] == exclude_case_id:
            continue
        results.append({
            "case_id":          meta["case_id"],
            "decision":         meta["decision"],
            "role":             meta["role"],
            "similarity_score": float(score),
            "case":             cases[meta["case_id"]],
        })
    return results[:k]

def stratify_results(results, max_per_class=5):
    buckets = defaultdict(list)
    for r in results:
        buckets[r["decision"]].append(r)
    stratified = []
    for decision in ["select", "reject"]:
        stratified.extend(buckets[decision][:max_per_class])
    return stratified

def build_evidence_chunks(stratified_results):
    evidence = []
    for r in stratified_results:
        c = r["case"]
        evidence.append({
            "case_id":          r["case_id"],
            "decision":         r["decision"],
            "similarity_score": r["similarity_score"],
            "role":             c["role"],
            "skills_cv":        c["entities"]["skills_cv"],
            "skills_jd":        c["entities"]["skills_jd"],
            "degrees":          c["entities"]["degrees"],
            "certifications":   c["entities"]["certifications"],
            "decision_reason":  c["decision_reason"],
        })
    return evidence

# -------------------------------
# Prompt builder
# -------------------------------
def _format_exemplar(chunk: dict, index: int) -> str:
    outcome   = chunk["decision"].upper()
    score     = chunk["similarity_score"]
    skills_cv = ", ".join(chunk["skills_cv"]) or "none listed"
    skills_jd = ", ".join(chunk["skills_jd"]) or "none listed"
    missing   = sorted(set(chunk["skills_jd"]) - set(chunk["skills_cv"]))
    missing_s = ", ".join(missing) or "none"
    degrees   = ", ".join(chunk["degrees"])   or "none listed"
    certs     = ", ".join(chunk["certifications"]) or "none listed"
    reason    = chunk["decision_reason"] or "no reason recorded"

    return f"""
### Historical Case {index} — Outcome: {outcome} (similarity: {score:.3f})
- Role: {chunk['role']}
- Candidate skills: {skills_cv}
- Job-required skills: {skills_jd}
- Skill gaps: {missing_s}
- Degrees: {degrees}
- Certifications: {certs}
- Decision rationale: {reason}
"""

def build_prompt(
    role: str,
    cv_text: str,
    jd_text: str,
    skills_cv: list,
    skills_jd: list,
    evidence_chunks: list,
) -> str:
    missing   = sorted(set(skills_jd) - set(skills_cv))
    missing_s = ", ".join(missing) or "none"
    exemplars = "".join(_format_exemplar(c, i) for i, c in enumerate(evidence_chunks, 1))

    return f"""You are an expert HR analyst making data-driven hiring decisions.
Evaluate the candidate's CV against the job description using the historical reference cases below.

## JOB DESCRIPTION
{jd_text}

## CANDIDATE PROFILE
- Role applied for: {role}
- Resume: {cv_text[:1500]}{"..." if len(cv_text) > 1500 else ""}
- Extracted skills: {", ".join(skills_cv) or "none detected"}
- Job-required skills: {", ".join(skills_jd) or "none detected"}
- Skill gaps: {missing_s}

## HISTORICAL REFERENCE CASES (vector-retrieved, stratified by outcome)
{exemplars}

## YOUR TASK
Using the job description, candidate profile, and historical cases as reference:

1. DECISION: Should this candidate be SELECTED or REJECTED?
2. CONFIDENCE: Rate your confidence (HIGH / MEDIUM / LOW)
3. KEY STRENGTHS: 2–3 strengths supporting selection
4. KEY GAPS: 2–3 gaps or concerns
5. REASONING: 2–3 sentences comparing the candidate to the historical cases

Respond in this exact JSON format only — no extra text:
{{
    "decision": "select" or "reject",
    "confidence": "high" or "medium" or "low",
    "key_strengths": ["strength1", "strength2"],
    "key_gaps": ["gap1", "gap2"],
    "reasoning": "Your explanation here"
}}

JSON Response:"""

# -------------------------------
# LLM call
# -------------------------------
def call_llm(prompt: str, url: str = OLLAMA_URL, model: str = MODEL_NAME) -> str:
    try:
        response = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"LLM Error: {e}"

# -------------------------------
# Response parser
# -------------------------------
def parse_decision(llm_output: str) -> dict[str, Any]:
    try:
        start = llm_output.find("{")
        end   = llm_output.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(llm_output[start:end])
    except Exception:
        pass

    # Fallback: extract decision from plain text
    lower = llm_output.lower()
    decision = "reject" if "reject" in lower else "select" if "select" in lower else "unknown"
    return {
        "decision":     decision,
        "confidence":   "low",
        "key_strengths": [],
        "key_gaps":      [],
        "reasoning":    llm_output[:500],
        "parse_error":  True,
    }

# -------------------------------
# Main entry point
# -------------------------------
def decide(
    role: str,
    cv_text: str,
    jd_text: str,
    skills_cv: list,
    skills_jd: list,
    *,
    k: int = 20,
    max_per_class: int = 5,
    exclude_case_id: str = None,
) -> dict[str, Any]:
    """
    Full pipeline: embed → retrieve → stratify → prompt → LLM → parse.
    Pass exclude_case_id when evaluating to prevent the query case from
    appearing in its own evidence (leave-one-out).
    Returns the structured decision dict.
    """
    index, metadata, cases = _load_artifacts()
    model = _load_model()

    query_vector = embed_query(model, role, cv_text, jd_text, skills_cv, skills_jd)
    retrieved    = retrieve_similar_cases(index, metadata, cases, query_vector, k=k,
                                          exclude_case_id=exclude_case_id)
    stratified   = stratify_results(retrieved, max_per_class=max_per_class)
    evidence     = build_evidence_chunks(stratified)

    prompt     = build_prompt(role, cv_text, jd_text, skills_cv, skills_jd, evidence)
    llm_output = call_llm(prompt)
    result     = parse_decision(llm_output)

    result["num_exemplars"]   = len(evidence)
    result["retrieval_mode"]  = "vector_faiss_two_view"
    result["raw_llm_output"]  = llm_output
    return result


# -------------------------------
# Script entry point (smoke test)
# -------------------------------
if __name__ == "__main__":
    role     = "data scientist"
    cv_text  = "experienced in python, machine learning, sql, data analysis and statistical modelling"
    jd_text  = "looking for a data scientist with python, machine learning, and cloud computing experience"

    skills_cv = ["python", "machine learning", "sql", "data analysis", "statistics"]
    skills_jd = ["python", "machine learning", "cloud computing", "data analysis"]

    print("Running LLM decision engine...")
    result = decide(role, cv_text, jd_text, skills_cv, skills_jd)

    print("\n=== DECISION ===")
    print(f"  Decision:    {result.get('decision', '?').upper()}")
    print(f"  Confidence:  {result.get('confidence', '?').upper()}")
    print(f"  Strengths:   {result.get('key_strengths', [])}")
    print(f"  Gaps:        {result.get('key_gaps', [])}")
    print(f"  Reasoning:   {result.get('reasoning', '')}")
    print(f"  Exemplars:   {result.get('num_exemplars', 0)}")
    if result.get("parse_error"):
        print("\n  [Warning] JSON parse failed — raw output shown in 'reasoning'")
