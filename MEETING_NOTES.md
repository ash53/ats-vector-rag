# Project Meeting Notes

_Date: 2026-05-12_

---

## Project Goal

Build an intelligent ATS (Applicant Tracking System) that:

- Matches CVs to job descriptions by **semantic meaning**, not keyword matching
- Benchmarks against a traditional keyword-based ATS
- Ensures fairness (no gender/cultural bias)
- Provides explainability
- Generates LLM-based rejection feedback for candidates

**Dataset:** Kaggle AI Recruitment Pipeline Dataset (~10,000+ CV–JD–Decision records)

---

## How the Pipeline Works

### Step 1 — Data Cleaning (`step1_cases.py`)

Reads the raw CSV dataset. Cleans and normalizes each row (CV, job description, hiring decision) into a structured case object.

```
dataset.csv  →  cases_stage1.jsonl
```

### Step 2 — Entity Extraction (`step2_entities.py`)

Extracts structured information from each case using vocabulary lists and regex:

- Skills from the CV and job description
- Degrees (bachelor, master, PhD, etc.)
- Certifications

```
cases_stage1.jsonl  →  cases_stage2.jsonl
```

### Step 3 — Vector Embeddings + Index (`step3_embeddings_faiss.py`)

Converts every case into number vectors (embeddings) and stores them in a FAISS search index. Uses a **two-view strategy**:

- **Case-level view** — role + CV + JD + decision reason (overall context)
- **Skill-alignment view** — candidate skills vs required skills + gaps

Both views are concatenated into one vector per case.

```
cases_stage2.jsonl  →  faiss_index.bin + faiss_metadata.json
```

### Step 4 — Retrieval (`step4_retrieval.py`)

Given a new CV + JD:

1. Builds the same two-view vector for the query
2. Searches FAISS for the 20 most similar historical cases
3. Balances results to 5 SELECT + 5 REJECT (so the LLM sees both outcomes)
4. Packages them into structured evidence chunks

### Step 5 — LLM Decision (`step5_llm_decision.py`)

Passes the 10 evidence chunks + the candidate's profile to the LLM with a structured prompt. The LLM returns:

```json
{
  "decision": "select" or "reject",
  "confidence": "high / medium / low",
  "key_strengths": ["..."],
  "key_gaps": ["..."],
  "reasoning": "..."
}
```

### Step 6 — Evaluation (`step6_evaluate.py`)

Runs the full pipeline on a random balanced sample of cases, compares predictions against ground truth labels, and reports:

- Accuracy, Precision, Recall, F1
- Confusion matrix
- Results saved to `eval_results.json`

---

## Full Pipeline Flow

```
dataset.csv
    ↓ step1 — clean & normalize
cases_stage1.jsonl
    ↓ step2 — extract entities
cases_stage2.jsonl
    ↓ step3 — embed & index (slow, runs on server)
faiss_index.bin
    ↓ step4 + step5 — retrieve + LLM decision
select / reject + reasoning
    ↓ step6 — evaluate
accuracy / F1 metrics
```

---

## Techniques Used

| Component          | Technique / Tool                                  |
| ------------------ | ------------------------------------------------- | -------------------------- |
| LLM                | llama3.1:8b via Ollama                            |
| Embedding model    | all-mpnet-base-v2 (sentence-transformers)         | try next: all-MiniLM-L6-v2 |
| Vector index       | FAISS IndexFlatIP (cosine similarity)             |
| Retrieval strategy | Two-view embedding (case-level + skill-alignment) |
| Evidence balancing | Stratified retrieval — 5 SELECT + 5 REJECT        |
| Entity extraction  | Rule-based (vocab matching + regex)               |

---

## Current Status

| Step                              | Status                                |
| --------------------------------- | ------------------------------------- |
| Step 1 — Data cleaning            | Done                                  |
| Step 2 — Entity extraction        | Done                                  |
| Step 3 — Embeddings + FAISS index | Running on uni server (in progress)   |
| Step 4 — Retrieval                | Done (code ready)                     |
| Step 5 — LLM decision engine      | Done (code ready)                     |
| Step 6 — Evaluation               | Done (code ready, waiting for Step 3) |

---

## Expected Accuracy

Based on comparable work (Amol's GraphRAG system using the same dataset):

- Amol's system: **40% accuracy, 57.1% F1** using llama3.2 (3B model)
- Sadia's system uses a better LLM (8B) and stronger embeddings
- **Estimated range: 45–60% accuracy**

Key limitation: the dataset itself has label noise (inconsistent ground truth decisions), which sets a natural ceiling for all approaches regardless of technique.

The model is expected to over-predict SELECT — high recall, lower precision. This is a known pattern with LLM-based hiring systems.

---

## What's Next — Roadmap

The goal is to compare multiple approaches and find which performs best. All approaches are evaluated using the same Step 6 script so results are directly comparable.

| #   | Approach                                                     | Purpose                                                                        |
| --- | ------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 1   | **Keyword baseline** (TF-IDF)                                | Required by proposal — establishes the floor score                             |
| 2   | **Single-view vs two-view ablation**                         | Validates the key design choice — does the skill-alignment view actually help? |
| 3   | **Embedding model swap** (MiniLM vs mpnet)                   | Tests quality vs speed tradeoff                                                |
| 4   | **Retrieval depth comparison** (3+3 vs 5+5 vs 8+8 exemplars) | Finds optimal amount of context for the LLM                                    |
| 5   | **LLM-based entity extraction**                              | Tests whether richer entities improve accuracy                                 |

---

## Final Deliverables (from proposal)

- Reproducible codebase with all approaches
- Evaluation results comparing all techniques
- Keyword baseline benchmark
- Fairness audit (bias testing)
- LLM-generated rejection feedback module
- Final report and presentation

---

## Scope for Improvement

### Still Required by the Proposal

These are core deliverables — not optional:

**1. Keyword Baseline**
No benchmark exists yet. Without it we can't claim the semantic approach is actually better than keyword matching. Should be built next. Approach: TF-IDF cosine similarity between CV and JD, threshold-based select/reject.

**2. Fairness Audit**
The proposal specifically requires testing for gender and cultural bias. Neither repo has touched this. Need to check whether the model systematically favours certain candidate profiles over others.

**3. Rejection Feedback Generator**
The proposal requires a structured 4-part rejection message:

1. Decision context (neutral summary of role requirements)
2. Specific skill gaps (2–3 concise gaps)
3. Actionable suggestions (courses, certifications, portfolio ideas)
4. Encouragement / invitation to re-apply

The LLM currently outputs `key_gaps` but doesn't generate a candidate-facing message. This builds directly on what's already there.

**4. Explainability**
SHAP or LIME to show _why_ a decision was made at the feature level. Currently the LLM gives plain-text reasoning but there's no quantitative breakdown of which factors drove the decision.

---

### Technical Improvements to the Pipeline

**5. LLM-Based Entity Extraction**
The current rule-based extractor only catches skills explicitly listed in `skills.txt`. A candidate who writes "built recommendation systems" would have that skill missed entirely. Replacing Step 2 with an LLM extractor would catch everything, including synonyms and implied skills.

**6. MMR Re-ranking**
Currently the top 10 evidence chunks are selected purely by similarity score. This risks retrieving near-identical cases. Adding Maximal Marginal Relevance (MMR) would diversify the evidence, giving the LLM a broader view of historical cases.

**7. JD Entity Extraction**
Step 2 extracts skills from CVs but job descriptions are also just raw text. Extracting structured requirements from JDs would allow more precise skill-gap calculation.

**8. Larger Evaluation Set**
20 test cases is too small — results won't be statistically reliable. Running on 100+ cases would give much more meaningful numbers.

---

### Bigger Architectural Ideas

**9. Hybrid Retrieval (Vector + Graph)**
Combining vector similarity AND graph-based entity overlap (like Amol's approach) into a single ranking could outperform either alone. The two signals are complementary — vectors capture semantic similarity, graphs capture explicit skill overlap.

**10. Fine-Tuned Embeddings**
`all-mpnet-base-v2` is a general-purpose model. Fine-tuning it on recruitment-specific data (CV–JD pairs with known outcomes) would make the embeddings more meaningful for this domain.

**11. Confidence Thresholding**
Currently every query gets a forced decision. Low-confidence cases should be flagged for human review rather than auto-decided. The `confidence` field from the LLM is already returned — just not acted on.

---

### Priority Order

| Priority | Task                               | Why                                                  |
| -------- | ---------------------------------- | ---------------------------------------------------- |
| 1        | Get eval results (wait for Step 3) | Need a number before anything else                   |
| 2        | Keyword baseline                   | Required by proposal, simple to build                |
| 3        | Rejection feedback generator       | Required by proposal, builds on what's already there |
| 4        | Fairness audit                     | Required by proposal                                 |
| 5        | Run roadmap ablations              | Compare approaches systematically                    |
| 6        | MMR + LLM entity extraction        | Improve retrieval quality                            |
| 7        | Explainability (SHAP/LIME)         | Required by proposal, complex to implement           |
| 8        | Hybrid retrieval                   | Ambitious, high potential payoff                     |
