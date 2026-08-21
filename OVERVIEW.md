> **Superseded.** This file describes the Phase 1 four-step pipeline as of
> 2026-05-11. The pipeline now has seven steps, the two-view embedding it
> describes has been removed as unjustified, and the project's findings have
> changed substantially. Kept as a record of where the work started.
>
> Current documents: [`README.md`](README.md) for how to run it,
> [`RESULTS.md`](RESULTS.md) for the numbers,
> [`EVALUATION_AND_APPROACH_PLAN.md`](EVALUATION_AND_APPROACH_PLAN.md) for
> interpretation and next steps.

# ATS Vector RAG — Project Overview

## What This Is

A **Phase 1 implementation** of a Vector-based Retrieval-Augmented Generation (RAG) pipeline for an Applicant Tracking System (ATS). The explicit goal of this phase is to build a clean, comparable vector retrieval baseline before contrasting it against a Graph-based RAG (GraphRAG) approach.

The system answers the question: *"Given a new CV and Job Description, what do similar historical hiring decisions tell us about this candidate?"*

---

## Dataset

- **Source**: [AI Recruitment Pipeline Dataset (Kaggle)](https://www.kaggle.com/datasets/yaswanthkumary/ai-recruitment-pipeline-dataset/data)
- **Contents**: Resumes, job descriptions, hiring decisions (`select` / `reject`), and human-written rationales
- **Expected location**: `data/raw/dataset.csv`

---

## Pipeline (4 Steps)

```
Raw CSV → Cases (JSONL) → Entities (JSONL) → FAISS Index → Retrieval + Evidence
  step1       step2           step3              step4
```

### Step 1 — Case Construction (`src/step1_cases.py`)
Reads the raw CSV and normalizes each row into a canonical `CV–JD–Decision` case object. Output is a JSONL file (`cases_stage1.jsonl`) with fields:

| Field | Description |
|---|---|
| `case_id` | Stable ID (`case_00001`, ...) |
| `role` | Job role (lowercased) |
| `cv_text` | Cleaned resume text |
| `jd_text` | Cleaned job description text |
| `decision` | `select` or `reject` |
| `decision_reason` | Human-written rationale |
| `metadata` | Source tag + interview transcript flag |

### Step 2 — Entity Extraction (`src/step2_entities.py`)
Enriches each case with structured entities extracted via vocabulary matching and regex patterns. No NLP model — purely rule-based for speed and reproducibility.

Extracts:
- **`skills_cv`** — candidate skills (matched against `data/vocab/skills.txt`)
- **`skills_jd`** — job-required skills (same vocab)
- **`degrees`** — degree keywords (bachelor, master, PhD, BSc, MSc, computer science, engineering, IT)
- **`certifications`** — matched against `data/vocab/certifications.txt`

Output: `cases_stage2.jsonl`

### Step 3 — Embeddings & FAISS Index (`src/step3_embeddings_faiss.py`)
Builds the vector store using **two complementary embedding views** per case, then concatenates them:

| View | What it captures | Analogue in GraphRAG |
|---|---|---|
| **Case-level** | Full CV + JD + decision reason | Overall case similarity |
| **Skill-alignment** | Candidate skills vs. required skills + gaps | Graph traversal over Skill nodes |

- **Model**: `all-mpnet-base-v2` (sentence-transformers, CPU)
- **Index**: `faiss.IndexFlatIP` — inner product on normalized embeddings = cosine similarity
- **Outputs**: `data/processed/faiss_index.bin` + `data/processed/faiss_metadata.json`

### Step 4 — Retrieval & Evidence Construction (`src/step4_retrieval.py`)
Given a new CV + JD:
1. Embeds the query using the same two-view strategy as Step 3
2. Retrieves top-k (default: 20) similar historical cases from FAISS
3. **Stratifies** results by decision — up to 5 `select` + 5 `reject` cases (balanced evidence)
4. Packages into structured **evidence chunks** ready for LLM prompting

Evidence chunk fields: `case_id`, `decision`, `similarity_score`, `role`, `skills_cv`, `skills_jd`, `degrees`, `certifications`, `decision_reason`

---

## Project Structure

```
ats-vector-rag/
├── src/
│   ├── step1_cases.py               # Normalize raw CSV → JSONL cases
│   ├── step2_entities.py            # Extract skills, degrees, certs
│   ├── step3_embeddings_faiss.py    # Build FAISS vector index
│   ├── step4_retrieval.py           # Retrieve similar cases + build evidence
│   └── utils.py                     # (placeholder — empty)
│
├── data/
│   ├── raw/                         # dataset.csv goes here (not in repo)
│   ├── processed/                   # Generated files (JSONL, FAISS index)
│   └── vocab/
│       ├── skills.txt               # ~100+ skill terms for matching
│       └── certifications.txt       # Certification terms for matching
│
├── vector_RAG.ipynb                 # Notebook (exploratory / demo)
├── requirements.txt                 # pandas, numpy, spacy, tqdm
├── README.md
└── OVERVIEW.md                      # ← You are here
```

---

## Dependencies

```
pandas       # CSV loading
numpy        # Embedding math
spacy        # (listed but unused in current steps — likely planned for Step 2 extension)
tqdm         # Progress bars
sentence-transformers  # all-mpnet-base-v2 (not in requirements.txt — must be added)
faiss-cpu    # Vector index (not in requirements.txt — must be added)
```

> **Note**: `sentence-transformers` and `faiss-cpu` are used in Steps 3–4 but are missing from `requirements.txt`. Add them before running.

---

## How to Run

```bash
# 1. Install dependencies (add missing packages first)
pip install -r requirements.txt
pip install sentence-transformers faiss-cpu

# 2. Place the Kaggle dataset
# data/raw/dataset.csv

# 3. Run the pipeline sequentially
python src/step1_cases.py
python src/step2_entities.py
python src/step3_embeddings_faiss.py
python src/step4_retrieval.py     # runs a smoke test on a sample data scientist query
```

---

## Design Decisions

- **Two-view embeddings**: Concatenating a semantic case view and a skill-alignment view is the key design choice — it simulates what GraphRAG achieves through explicit skill-node traversal, without a graph.
- **Rule-based entity extraction**: Vocabulary matching and regex over an NLP model — faster, deterministic, no GPU needed.
- **Stratified retrieval**: Equal-weight evidence from both `select` and `reject` cases prevents bias toward the majority class.
- **JSONL format**: Chosen for streaming, RAG-friendliness, and easy line-by-line processing.

---

## Context

This is **Phase 1** of a larger comparison study. The vector pipeline here will be benchmarked against a GraphRAG approach. The decision between the two approaches will drive the architecture of the final ATS reasoning system.
