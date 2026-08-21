> **Superseded.** This file describes the Phase 1 four-step pipeline as of
> 2026-05-11. The pipeline now has seven steps, the two-view embedding it
> describes has been removed as unjustified, and the project's findings have
> changed substantially. Kept as a record of where the work started.
>
> Current documents: [`README.md`](README.md) for how to run it,
> [`RESULTS.md`](RESULTS.md) for the numbers,
> [`EVALUATION_AND_APPROACH_PLAN.md`](EVALUATION_AND_APPROACH_PLAN.md) for
> interpretation and next steps.

# ATS Vector RAG — Roadmap & Approach Comparison Plan

## Goal
Try different approaches on the same pipeline to find which combination gives the best hiring decision accuracy. Each approach is measured against the same evaluation script (Step 6) using accuracy, precision, recall, and F1.

---

## Immediate Next Step

### Step 6 — Evaluation Script
Run the pipeline on a held-out sample of cases, compare predicted decisions against ground truth labels, and compute accuracy / precision / recall / F1. This is the yardstick for all comparisons below. Nothing can be compared without it.

---

## Approaches to Compare (in priority order)

### 1. Keyword Baseline
**What:** TF-IDF or BM25 matching between CV text and JD text — no embeddings, no LLM. Produces a similarity score and a threshold-based select/reject decision.

**Why:** The project proposal explicitly requires benchmarking against a traditional keyword-based ATS. This is the floor score to beat.

**Where:** New file `src/baseline_keyword.py`

---

### 2. Ablation — Two-View vs Single-View Embeddings
**What:** Compare the current two-view embedding (case-level + skill-alignment concatenated) against a single-view embedding (case-level only).

**Why:** The two-view approach is Sadia's key design choice. It needs to be validated — does the skill-alignment view actually improve retrieval quality, or does it add noise?

**Where:** Toggle in Step 3 / Step 5 (`embed_query` function)

---

### 3. Embedding Model Comparison
**What:** Swap `all-mpnet-base-v2` (current, 768-dim, slower) for `all-MiniLM-L6-v2` (384-dim, faster — same model Amol uses) and compare decision quality.

**Why:** Establishes the quality-vs-speed tradeoff between the two models. If MiniLM performs similarly, the pipeline becomes significantly faster.

**Where:** One-line change in `step3_embeddings_faiss.py` and `step5_llm_decision.py`

---

### 4. Retrieval Depth / Stratification Ratio
**What:** Test different values for how many exemplars are retrieved and passed to the LLM. Current: top-20 retrieved, 5 SELECT + 5 REJECT passed to LLM. Try: 3+3, 5+5 (current), 8+8.

**Why:** Too few exemplars gives the LLM insufficient context. Too many may dilute signal or exceed context limits.

**Where:** Parameters in `step5_llm_decision.py` (`k` and `max_per_class` in `decide()`)

---

### 5. LLM-Based Entity Extraction (Step 2 upgrade)
**What:** Replace rule-based vocab matching in Step 2 with an LLM call (same Ollama setup) to extract entities from free-form resume text.

**Why:** The current vocabulary lists miss skills not explicitly listed in `skills.txt`. An LLM extractor is more flexible, handles synonyms, and can also extract company names. Compare whether richer entities improve final decision accuracy.

**Where:** New file `src/step2_entities_llm.py` alongside the existing rule-based version

---

## Metrics to Track

For every approach, record:

| Metric | Description |
|---|---|
| Accuracy | % of correct select/reject decisions |
| Precision (SELECT) | Of predicted SELECT, how many were actually correct |
| Recall (SELECT) | Of actual SELECT cases, how many were found |
| F1 Score | Harmonic mean of precision and recall |
| Avg. inference time | Seconds per candidate |

---

## Results Table — moved

Results live in [`RESULTS.md`](RESULTS.md), generated from the raw result files.
The empty table below was never filled in and is kept only to show what was
planned versus what was actually measured.

## Results Table (original plan, never filled)

| Approach | Accuracy | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|
| Keyword baseline | — | — | — | — | TF-IDF threshold |
| Two-view FAISS + llama3.1:8b (current) | — | — | — | — | Baseline RAG |
| Single-view FAISS + llama3.1:8b | — | — | — | — | Ablation |
| MiniLM + llama3.1:8b | — | — | — | — | Faster model |
| Two-view + 3+3 exemplars | — | — | — | — | Less context |
| Two-view + 8+8 exemplars | — | — | — | — | More context |
| LLM entity extraction + llama3.1:8b | — | — | — | — | Richer entities |
