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
import torch
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
# Load embedding model
# -------------------------------
# all-mpnet-base-v2 provides strong semantic embeddings for
# resumes and job descriptions.
# Apple Silicon GPU (MPS) when available — roughly 3x faster than CPU here.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 64 if DEVICE == "mps" else 16
print(f"Embedding device: {DEVICE} (batch size {BATCH_SIZE})")

model = SentenceTransformer(
    "all-mpnet-base-v2",
    device=DEVICE
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

    Purpose:
    "Have we seen a similar CV–JD case before?"

    NOTE — leakage fix (2026-08-08): `decision_reason` was previously part of
    this text. That field states the outcome ("Lacked leadership skills...",
    "Strong technical skills..."), so indexing it meant retrieval was partly
    matching on the answer. A TF-IDF + logistic regression probe recovers the
    label from `decision_reason` alone with 91.9% accuracy vs. 58.2% from the
    resume, so the field is effectively the label in disguise.

    It was also an asymmetry bug: the query-side builders in Step 4 and Step 5
    never included this line, so indexed vectors and query vectors were built
    from different templates. Both sides now match exactly.
    """
    return f"""
    Role: {case['role']}
    Resume: {case['cv_text']}
    Job Description: {case['jd_text']}
    """

# -------------------------------
# Skill-alignment view — REMOVED 2026-08-21
# -------------------------------
# The index used to be [case view || skill view], 1536-d. The skill view listed
# candidate skills, required skills and explicit gaps, and was meant to mimic
# graph traversal over Skill nodes in GraphRAG.
#
# It was measured (src/exp_retrieval_views.py, 300 queries leave-one-out, k=10)
# and it does not pay for itself:
#
#   case only (768-d)     same-decision@10  56.6%
#   skill only (768-d)                      52.7%
#   two-view (1536-d)                       56.0%
#
# Paired t-test, two-view minus case-only: -0.6pp, p=0.50 — indistinguishable,
# while doubling the vector dimension, the index size (62 MB -> 31 MB) and the
# search cost. The skill view alone is significantly worse than the case view
# (p=0.007).
#
# The two-view index is preserved as data/processed/faiss_index_twoview.bin so
# that results produced before this change remain reproducible. Everything
# committed up to 0f75e4c used it; note the similarity scale also changes, since
# one normalised view scores in [0, 1] rather than [0, 2].

# -------------------------------
# Load cases and prepare embedding texts
# -------------------------------
case_texts = []
metadata = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Preparing embedding texts"):
        case = json.loads(line)

        # Build embedding input
        case_texts.append(build_case_text(case))

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
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

# -------------------------------
# Single case-level view
# -------------------------------
# Previously hstacked with a skill-alignment view; see the note above for why
# that was dropped. Vectors are already L2-normalised by the encoder, so inner
# product is cosine similarity.
embeddings = np.asarray(case_embeddings, dtype="float32")

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
