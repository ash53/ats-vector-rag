"""
STEP 3 — Embeddings & Vector Store (FAISS)

This script builds a vector-based retrieval index over historical
CV-JD-Decision cases using sentence embeddings and FAISS.

The resulting index enables case-based retrieval for vectorized RAG,
serving as a fair comparison point to GraphRAG subgraph retrieval.
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
from tqdm import tqdm

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------
# File paths
# -------------------------------
INPUT_PATH = Path("data/processed/cases_stage2.jsonl")
INDEX_PATH = Path("data/processed/faiss_index.bin")
META_PATH = Path("data/processed/faiss_metadata.json")

# Ensure previous steps were executed
assert INPUT_PATH.exists(), "cases_stage2.jsonl not found. Run Step 1 and Step 2 first."

# -------------------------------
# Load embedding model (CPU only)
# -------------------------------
# all-mpnet-base-v2 provides strong semantic embeddings for
# resumes, job descriptions, and decision rationales.
model = SentenceTransformer(
    "all-mpnet-base-v2",
    device="cpu"
)

# -------------------------------
# Helper: Build case-level embedding text
# -------------------------------
def build_case_text(case):
    """
    Constructs a full-context textual representation of a recruitment case.

    This embedding captures:
    - Role
    - Resume content
    - Job description content
    - Decision rationale

    Purpose:
    "Have we seen a similar CV–JD–Decision case before?"
    """
    return f"""
    Role: {case['role']}
    Resume: {case['cv_text']}
    Job Description: {case['jd_text']}
    Decision Reason: {case['decision_reason']}
    """

# -------------------------------
# Helper: Build skill-alignment embedding text
# -------------------------------
def build_skill_text(case):
    """
    Constructs a skill-focused representation highlighting:
    - Candidate skills
    - Job-required skills
    - Explicit skill gaps

    This mimics graph traversal over Skill nodes in GraphRAG.
    """
    skills_cv = ", ".join(case["entities"]["skills_cv"])
    skills_jd = ", ".join(case["entities"]["skills_jd"])

    missing_skills = sorted(
        set(case["entities"]["skills_jd"]) - set(case["entities"]["skills_cv"])
    )
    missing_text = ", ".join(missing_skills)

    return f"""
    Candidate skills: {skills_cv}
    Job required skills: {skills_jd}
    Missing skills: {missing_text}
    """

# -------------------------------
# Load cases and prepare embedding texts
# -------------------------------
case_texts = []
skill_texts = []
metadata = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Preparing embedding texts"):
        case = json.loads(line)

        # Build embedding inputs
        case_texts.append(build_case_text(case))
        skill_texts.append(build_skill_text(case))

        # Lightweight metadata for retrieved cases
        metadata.append({
            "case_id": case["case_id"],
            "decision": case["decision"],
            "role": case["role"]
        })

# -------------------------------
# Encode texts into embeddings
# -------------------------------
# normalize_embeddings=True allows cosine similarity
# via inner product in FAISS.
print("Encoding case-level texts...")
case_embeddings = model.encode(
    case_texts,
    batch_size=8,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Encoding skill-alignment texts...")
skill_embeddings = model.encode(
    skill_texts,
    batch_size=8,
    show_progress_bar=True,
    normalize_embeddings=True
)

# -------------------------------
# Combine embeddings
# -------------------------------
# Concatenation preserves two complementary semantic views:
# 1. Global CV–JD–Decision similarity
# 2. Explicit skill overlap and gaps
embeddings = np.hstack([case_embeddings, skill_embeddings]).astype("float32")

# -------------------------------
# Build FAISS index
# -------------------------------
# Using IndexFlatIP because embeddings are normalized,
# so inner product == cosine similarity.
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)

# Add embeddings to the index
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors of dimension {dim}.")

# -------------------------------
# Persist index and metadata
# -------------------------------
faiss.write_index(index, str(INDEX_PATH))

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("FAISS index and metadata successfully saved.")
