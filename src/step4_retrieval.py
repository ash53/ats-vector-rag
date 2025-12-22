"""
STEP 4 — Retrieval & Evidence Construction (Vector RAG)

This script performs similarity-based retrieval over historical
CV-JD-Decision cases using a FAISS vector index.

Given a new CV and Job Description, it:
1. Embeds the input using the same logic as Step 3
2. Retrieves the top-k most similar historical cases
3. Stratifies results by hiring outcome (select / reject)
4. Constructs structured, LLM-ready evidence chunks

This is the vector-based analogue of subgraph retrieval in GraphRAG.
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
from pathlib import Path
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------
# File paths
# -------------------------------
INDEX_PATH = Path("data/processed/faiss_index.bin")
META_PATH = Path("data/processed/faiss_metadata.json")
CASES_PATH = Path("data/processed/cases_stage2.jsonl")

assert INDEX_PATH.exists(), "FAISS index not found. Run Step 3 first."
assert META_PATH.exists(), "FAISS metadata not found. Run Step 3 first."
assert CASES_PATH.exists(), "cases_stage2.jsonl not found. Run Step 2 first."

# -------------------------------
# Load FAISS index
# -------------------------------
index = faiss.read_index(str(INDEX_PATH))

# -------------------------------
# Load metadata
# -------------------------------
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# -------------------------------
# Load full cases (needed for evidence)
# -------------------------------
cases = {}
with open(CASES_PATH, "r", encoding="utf-8") as f:
    for line in f:
        case = json.loads(line)
        cases[case["case_id"]] = case

# -------------------------------
# Load embedding model (must match Step 3)
# -------------------------------
model = SentenceTransformer(
    "all-mpnet-base-v2",
    device="cpu"
)

# -------------------------------
# Build embedding texts (same logic as Step 3)
# -------------------------------
def build_case_text(role, cv_text, jd_text):
    """
    Builds the case-level semantic representation for the query.
    """
    return f"""
    Role: {role}
    Resume: {cv_text}
    Job Description: {jd_text}
    """

def build_skill_text(skills_cv, skills_jd):
    """
    Builds the skill-alignment representation for the query.
    """
    missing = sorted(set(skills_jd) - set(skills_cv))
    return f"""
    Candidate skills: {", ".join(skills_cv)}
    Job required skills: {", ".join(skills_jd)}
    Missing skills: {", ".join(missing)}
    """

# -------------------------------
# Embed query (CV + JD)
# -------------------------------
def embed_query(role, cv_text, jd_text, skills_cv, skills_jd):
    """
    Encodes a new CV–JD pair into a single query vector,
    using the same multi-view embedding strategy as Step 3.
    """
    case_text = build_case_text(role, cv_text, jd_text)
    skill_text = build_skill_text(skills_cv, skills_jd)

    case_emb = model.encode(
        [case_text],
        normalize_embeddings=True
    )

    skill_emb = model.encode(
        [skill_text],
        normalize_embeddings=True
    )

    return np.hstack([case_emb, skill_emb]).astype("float32")

# -------------------------------
# Retrieve top-k similar cases
# -------------------------------
def retrieve_similar_cases(query_vector, k=20):
    """
    Retrieves the top-k most similar historical cases
    from the FAISS index.
    """
    scores, indices = index.search(query_vector, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        meta = metadata[idx]
        case = cases[meta["case_id"]]

        results.append({
            "case_id": meta["case_id"],
            "decision": meta["decision"],
            "role": meta["role"],
            "similarity_score": float(score),
            "case": case
        })

    return results

# -------------------------------
# Stratify results by decision
# -------------------------------
def stratify_results(results, max_per_class=5):
    """
    Ensures balanced evidence by selecting both
    positive (select) and negative (reject) cases.
    """
    buckets = defaultdict(list)
    for r in results:
        buckets[r["decision"]].append(r)

    stratified = []
    for decision in ["select", "reject"]:
        stratified.extend(buckets[decision][:max_per_class])

    return stratified

# -------------------------------
# Construct LLM-ready evidence chunks
# -------------------------------
def build_evidence_chunks(stratified_results):
    """
    Converts retrieved cases into structured evidence chunks
    suitable for LLM prompting.
    """
    evidence = []

    for r in stratified_results:
        c = r["case"]

        chunk = {
            "case_id": r["case_id"],
            "decision": r["decision"],
            "similarity_score": r["similarity_score"],
            "role": c["role"],
            "skills_cv": c["entities"]["skills_cv"],
            "skills_jd": c["entities"]["skills_jd"],
            "degrees": c["entities"]["degrees"],
            "certifications": c["entities"]["certifications"],
            "decision_reason": c["decision_reason"]
        }

        evidence.append(chunk)

    return evidence

# -------------------------------
# Example run (smoke test)
# -------------------------------
if __name__ == "__main__":
    # Example input (replace with real input later)
    role = "data scientist"
    cv_text = "experienced in python, machine learning, sql"
    jd_text = "looking for data scientist with python, machine learning, cloud experience"

    skills_cv = ["python", "machine learning", "sql"]
    skills_jd = ["python", "machine learning", "cloud computing"]

    query_vector = embed_query(
        role, cv_text, jd_text, skills_cv, skills_jd
    )

    retrieved = retrieve_similar_cases(query_vector, k=20)
    stratified = stratify_results(retrieved, max_per_class=5)
    evidence = build_evidence_chunks(stratified)

    print("\nRetrieved Evidence:\n")
    for e in evidence:
        print(
            f"[{e['decision'].upper()} | score={e['similarity_score']:.3f}] "
            f"{e['decision_reason']}"
        )
