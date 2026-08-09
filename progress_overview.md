# Group Project — Progress Overview
*Last updated: 2026-08-09 (previous: 2026-05-11)*

Cross-repo status. For evaluation strategy, measured results in full, and the
ranked plan, see `EVALUATION_AND_APPROACH_PLAN.md` — that document is
canonical for anything methodological.

---

## Project Goal (from Proposal)

Build an **intelligent and fair ATS** that:
1. Matches CVs to job descriptions by **semantic meaning**, not keyword overlap
2. **Benchmarks** against a traditional keyword-based ATS
3. Audits and improves **fairness** (gender/cultural bias)
4. Provides **explainability**
5. Generates **LLM-based rejection feedback**

**Dataset:** Kaggle AI Recruitment Pipeline (10,174 CV–JD–Decision rows, 45 roles, 50/50 class balance)

---

## The headline finding

**The dataset's labels are largely unlearnable from the candidate data, and this caps every approach either of us can build.**

Measured, not assumed:

- A **fully supervised** TF-IDF + logistic regression classifier — trained on all 10,174 rows *with* the labels — reaches **58.2%** accuracy (AUC 0.639) from the resume. That is the ceiling, not 100%.
- The **reason for each decision is uncorrelated with the candidate's data**: predicting which of the top-10 reasons was given scores 10.5% against a 10.4% majority baseline. Candidates rejected for "lacking cloud experience" mention cloud at 29.4%, against a corpus base rate of 33.1%.
- The **job descriptions are nearly empty**: mean 53 words, only 21.7% state any requirement. Typical JD: *"We're hiring a E-commerce Specialist to develop and deliver high-quality solutions to transform our healthcare."*

Consequence: every method tried by either of us lands in a **47.7%–55.3%** band, and decision accuracy against this label cannot be the headline metric. The project's research question has shifted to *how you evaluate an LLM hiring system when the ground truth is unreliable, and whether these systems are fair.*

---

## Where We Are Against the Timeline

| Phase | Planned | Status |
|---|---|---|
| Literature Review | wk 1–2 | Complete |
| Data Preparation | wk 3–4 | **Done** — both repos |
| Model Development | wk 5–8 | **Done** — GraphRAG, two vector RAG pipelines, fine-tuning attempted |
| Evaluation & Fairness | wk 9–10 | **Evaluation substantially done** (Sadia); **fairness not started** |
| LLM Feedback Integration | wk 10–11 | **Not started** — decisions exist, candidate-facing feedback does not |
| Reporting & Presentation | wk 12 | Not started |

---

## Sadia — `ats-vector-rag`

Six-step pipeline, fully evaluated. Branch `eval-harness-and-leak-fix` (7 commits).

**Pipeline:** case construction → rule-based entity extraction → two-view embeddings (case-level + skill-alignment, `all-mpnet-base-v2`, concatenated) → FAISS IndexFlatIP → stratified retrieval (5 select + 5 reject) → `llama3.1:8b` decision → evaluation.

### Results, n=300, seed 42, temperature 0

| System | Accuracy | 95% CI | F1 | Select rate |
|---|---|---|---|---|
| **RAG pipeline** | **55.3%** | 49.5–61.0% | 66.3% | 82.7% |
| Supervised TF-IDF+LogReg (ceiling) | 53.7% | 47.8–59.4% | 52.2% | 47.0% |
| always-select (degenerate) | 50.0% | 44.2–55.8% | 66.7% | 100% |
| Keyword TF-IDF (traditional ATS) | 47.7% | 41.9–53.5% | 39.9% | 37.0% |

Paired McNemar: beats always-select (p=0.037), **indistinguishable from bag-of-words logistic regression** (p=0.75). The keyword baseline is **at chance** (AUC 0.474) — there is nothing in a 53-word JD to keyword-match against.

### Fixed along the way

- **Label leak in Step 3.** `decision_reason` was embedded into every indexed vector; that field states the outcome and a classifier recovers the label from it alone at 91.9%. Removed, index rebuilt. It was also an asymmetry bug — the query side never included that line, so indexed and query vectors came from different templates.
- **Unreproducible runs.** Ollama defaults to temperature 0.8; the identical prompt on the identical 40 cases scored 45.0% then 50.0%. Now pinned to greedy with a fixed seed.
- **Per-call reloading.** Step 5 re-read the 62 MB index, the 38 MB case file and the embedding model on every case. Cached.
- **Index rebuild time:** ~12 h → 11 min by moving encoding to MPS.

### Other measured findings

- **Confidence is uninformative** — "high" on 276/300 cases at 56% accuracy, "medium" on 24 at 50%. Abstention has nothing to threshold on.
- **The select bias is fixable but doesn't buy accuracy.** Prompt changes moved the select rate from 90% to 47.5% with no accuracy gain — the signature of a near-random underlying ranking.
- **The model breaks its own stated rules on 30% of cases** — given an explicit aggregation rule and its own counts, it decided otherwise 12 times in 40. Moving the arithmetic into code recovered 5 points.
- **Retrieval is role-matching and little else** — the top-20 are 20/20 the same role, spanning 0.064 of similarity out of a 2.0 maximum.

### Running now

Four controls at n=300 isolating whether retrieval contributes anything: retrieved vs. random-same-role vs. random-any-role vs. zero-shot, prompt held fixed. **This is the experiment the write-up hinges on.**

### Built, awaiting use

Blind gold-set labelling tooling — 150 cases, one self-contained HTML sheet per annotator, different order each, 10 hidden repeat probes for the human noise floor, plus a scorer for inter-annotator kappa and system-vs-human agreement.

---

## Amol — `ats-graph-rag`

Last pulled 2026-08-08 (read-only; nothing committed or pushed there). Three commits since May, latest `77d75d9` "New files regarding new approach" (2026-07-27).

**Approach 1 — GraphRAG:** Llama 3.2 entity extraction, NetworkX graph (821 nodes / 1,336 edges over 100 candidates), weighted Jaccard retrieval (skills 50%, degrees 20%, certs 20%, companies 10%). **40% accuracy, 57.1% F1, 80.0% recall on 10 test cases.**

**Approach 2 — Vector RAG:** packaged module, MiniLM + ChromaDB + MMR, leave-one-out support, Precision@K / NDCG@K / MRR implemented.

**Approach 3 — Resume2Vec fine-tuning** (new): contrastive fine-tuning of MiniLM, TripletLoss, 12,138 training examples, 3 epochs. Data sufficiency passed (45 roles, 478k potential pairs). Baseline logged: accuracy@1 0.467, NDCG@10 0.438, MRR@10 0.626, same-decision retrieval 55.2%. **Execution stops at the training cell (19 of 39)** — no fine-tuned evaluation yet, so there is no before/after.

**Approach 4 — Resume vs. resume+interview** (new): summary file reports 30% → 40% (+10pp) on 10 candidates, but the notebook has **no saved outputs** (0/12 code cells executed as committed) and `rag_comparison_summary.json` is **truncated mid-write**. Not yet evidence. Worth settling on ≥300 cases — transcripts are the strongest single field (61.2% supervised vs 58.2% for the resume) — but note a screening ATS would not have interview data.

**Also present:** `debug-rag-comparison.ipynb`, diagnosing a run where both systems scored 0%.

---

## Convergent evidence across both repos

Two independently built pipelines — different embedding models, vector stores, retrieval strategies and LLMs — produce the same degenerate behaviour:

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Amol GraphRAG (n=10) | 40.0% | 44.4% | **80.0%** | 57.1% |
| Sadia vector RAG (n=40) | 45.0% | 47.1% | **80.0%** | 59.3% |

Recall identical to the decimal. That rules out an implementation bug in either repo — the cause is upstream of both: empty JDs, near-identical exemplars, and a decision layer with no cost for a false positive.

---

## Status vs. the Proposal

| Requirement | Status |
|---|---|
| Semantic CV–JD matching | **Done** — both repos |
| Benchmark vs. keyword ATS | **Done** — keyword TF-IDF at 47.7%, AUC 0.474 (at chance) |
| Full-dataset evaluation | **Done** — n=300 with confidence intervals and paired significance tests |
| Fairness testing | **Not started** — tooling designed (name-swap matched pairs), not built |
| Explainability (SHAP/LIME) | **Not started** |
| Rejection feedback generator | **Not started** — `key_gaps` exists, the candidate-facing message does not |
| Reproducible codebase | **Done** on Sadia's side — seeded, temperature-pinned, results tracked in git |
| Demo | Not started |

---

## Next Steps

1. **Both: label the 150-case gold set.** Independently and blind, ~2–3 h each. Every automated reference line sits in a 7-point band, so human agreement is the only measurement left that separates a good system from a lucky one. Sheets are in `data/gold/` (this repo).
2. **Sadia: finish the retrieval controls**, then build the counterfactual + name-swap harness — objective correctness without ground truth, plus the fairness deliverable, from one piece of code.
3. **Sadia: validate the two-view embedding**, her own key design choice, at the retrieval layer only.
4. **Amol: finish the fine-tune** (20 cells from a before/after) and **re-run the transcript comparison** on ≥300 cases with outputs saved.
5. **Both: rejection feedback generator** — cheapest remaining proposal deliverable.
6. **Raise with the supervisor:** this dataset cannot support a "which retrieval approach wins" leaderboard. The project is stronger framed as evaluation methodology plus fairness. Consider a second dataset with real outcomes.

**Deprioritised:** cross-encoder reranking, hybrid retrieval, further embedding work. With a mid-50s ceiling and a near-random within-role ranking, better retrieval has nothing to bite on — and the running controls may show it contributes nothing at all.

---

## Housekeeping

- This file lives in `ats-vector-rag` (moved 2026-08-09 from the unversioned `Group Project/` top level, where it was invisible to anyone else). It covers **both** repos, so it needs updating when either side moves — not only this one.
- Amol's `PROJECT_OVERVIEW.md` is untracked in his repo; Sadia's work is on branch `eval-harness-and-leak-fix`, not yet merged to `main`.
- Neither repo shares a metrics module or a frozen test split with the other. Everything in Sadia's repo now uses seed 42 and `data/processed/eval_*.json`; adopting the same on Amol's side is what would make the two comparable.
