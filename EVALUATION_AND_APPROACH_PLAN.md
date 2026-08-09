# Evaluation Strategy & Next Approaches

*Written 2026-08-08, updated 2026-08-09 with measured results. Responds to Amol's question: "accuracy ain't good — there should be something other than accuracy to test outputs of an LLM."*

---

## 0. The finding that reframes everything

Amol is right that accuracy is the wrong metric, but for a bigger reason than "the LLM is imperfect". Measured directly on the dataset; scripts reproducible in ~2 min.

### 0.1 The learnable ceiling is ~58%, not 100%

A fully supervised TF-IDF + logistic regression classifier, trained on **all 10,174 rows** with 5-fold cross-validation — the strongest thing you can do with these features, with direct access to the labels, no LLM involved:

| Input field | Accuracy | ROC-AUC |
|---|---|---|
| Resume only | **58.2%** | 0.639 |
| Resume + Job Description | 57.3% | 0.631 |
| Job Description only | 54.7% | 0.592 |
| Role only | 49.8% | 0.504 |
| Transcript only | **61.2%** | 0.693 |
| Reason_for_decision | 91.9% | 0.987 |

Class balance is 50/50 (5,060 select / 5,114 reject), so random = 50%. A supervised model that has *seen the labels* extracts only 8 points of signal from the resume.

The `Reason_for_decision` row is a **label leak**, not a result — the reason text states the outcome. It must never enter any embedding, prompt, or index.

> **Status: fixed.** Step 3 was embedding `decision_reason` into every indexed vector. Removed, index rebuilt (commit `c0f0e64`). It was also an asymmetry bug — Steps 4 and 5 never included that line, so indexed and query vectors came from different templates.

### 0.2 The decision rationale is not grounded in the candidate's data

The top 10 reason strings cover 69% of the dataset (7,000 rows) — a clean 10-class problem. Predicting *which* reason was given:

| Input | Reason-class accuracy | Majority baseline |
|---|---|---|
| Resume | 10.5% | 10.4% |
| Transcript | 10.4% | 10.4% |
| Resume + Transcript | 10.3% | 10.4% |

**Zero signal.** Groundedness check — for candidates rejected *specifically* for lacking a skill, does their resume mention it?

| Rejection reason | Skill appears in resume | Corpus base rate |
|---|---|---|
| "Lacks hands-on experience with cloud platforms" | 29.4% | 33.1% |
| "Lacked leadership skills for a senior position" | 9.8% | 10.6% |
| "Insufficient system design expertise" | 2.3% | 3.4% |
| "No experience in back-end development" | 1.4% | 2.2% |

Candidates rejected for lacking cloud experience mention cloud at *the same rate as everyone else*. The labels are synthetic and largely uncorrelated with the content they claim to describe.

### 0.3 The job descriptions carry almost no requirements

- 3,446 unique JDs, mean length **53 words**
- Only **21.7%** contain any explicit requirement phrase
- Typical JD: *"We're hiring a E-commerce Specialist to develop and deliver high-quality solutions to transform our healthcare."*

The JD is essentially the role title plus marketing copy. Confirmed downstream: asked to extract 4–6 requirements from a JD, the model could only find two, because there were only two.

### 0.4 What this means

Stop optimising decision accuracy against this label. The research question is:

> **"How do you evaluate an LLM hiring system when the ground-truth labels are unreliable — and which approach produces the most faithful, stable, and fair decisions?"**

---

## 1. Measured results

All runs: seed 42, leak-free index, `llama3.1:8b` via Ollama, identical stratified test samples.

### 1.1 Headline comparison, n=300

| System | Accuracy | 95% CI | Precision | Recall | F1 | Select rate |
|---|---|---|---|---|---|---|
| **RAG: two-view FAISS + mpnet + llama3.1:8b** | **55.3%** | 49.5–61.0% | 53.2% | 88.0% | 66.3% | 82.7% |
| Supervised TF-IDF + LogReg (ceiling) | 53.7% | 47.8–59.4% | 53.9% | 50.7% | 52.2% | 47.0% |
| always-select (degenerate) | 50.0% | 44.2–55.8% | 50.0% | 100% | **66.7%** | 100% |
| Keyword TF-IDF (the "traditional ATS") | 47.7% | 41.9–53.5% | 46.9% | 34.7% | 39.9% | 37.0% |

RAG confusion matrix: TP 132, FP 116, FN 18, **TN 34**. Zero dropped cases, zero parse failures.

Paired McNemar tests on the same 300 cases:

| Comparison | RAG only right | Other only right | p |
|---|---|---|---|
| vs always-select | 34 | 18 | **0.037** |
| vs keyword TF-IDF | 90 | 67 | 0.079 |
| vs supervised LogReg | 79 | 74 | 0.75 |

**Read honestly:** a real but small effect at the edge of detectability. Against a fixed 50% the binomial gives p=0.073 (not significant); against the actual always-select predictions the paired test gives p=0.037 (significant, and more sensitive). The pipeline modestly beats the degenerate baseline and is **statistically indistinguishable from a bag-of-words classifier that runs in under a second**.

**The keyword baseline is at chance** (AUC 0.474). CV↔JD cosine similarity carries no information about the decision on this dataset — the direct consequence of §0.3. Note what this does to the proposal's framing: "semantic matching beats keyword matching" becomes "keyword matching is at chance, and so is nearly everything else."

The competitive band across every method tried is **47.7% to 55.3%**.

### 1.2 Reproducibility — fixed

Ollama defaults to temperature 0.8. The identical prompt on the identical 40 cases scored **45.0%** and then **50.0%** — a 5-point swing from sampling alone, larger than most differences we were trying to detect.

Consequence: the n=40 numbers cannot rank anything, and an earlier conclusion drawn from them ("the pipeline is worse than always-select") is **wrong at n=300**. `call_llm` now sends `temperature=0` with a fixed seed (commit `bd04adb`); sampling remains available for self-consistency, where variance is the measurement.

### 1.3 Prompt variants, n=40 — ranking superseded, mechanics valid

| Variant | Acc | F1 | Select rate |
|---|---|---|---|
| v0 baseline | 50.0% | 64.3% | 90.0% |
| v1 reason before deciding | 47.5% | 57.1% | 72.5% |
| v2 base rate + no default | 42.5% | 56.6% | 82.5% |
| v3 requirement checklist | 42.5% | 41.0% | 47.5% |
| v3-rule (aggregation in code) | 47.5% | 57.1% | 72.5% |

Ranking is untrustworthy — n=40, temperature 0.8. What survives:

- **The select bias is fixable but doesn't buy accuracy.** Select rate moved 90% → 47.5%; accuracy did not improve. If the model had real signal, dragging its operating point would have traded a few false positives for many fewer errors. It swapped false positives for false negatives roughly one-for-one — the signature of a near-random ranking. Consistent with keyword AUC 0.474 and supervised AUC 0.601.
- **The model breaks its own stated rule on 30% of cases.** Given "reject if fewer than half the requirements are met" and its own counts, it decided otherwise 12 times in 40. Moving that arithmetic into code recovered 5 points. **Do not let the LLM aggregate.**
- **Only 9 of 40 cases were right under all five variants; 11 were wrong under all five.** The other 20 flip on prompt wording alone.

### 1.4 Confidence is uninformative

| Confidence | n | Accuracy |
|---|---|---|
| high | 276 | 56% |
| medium | 24 | 50% |

Flat, at n=300. Confidence-based abstention — "escalate the uncertain ones to a human" — has nothing to threshold on as currently prompted.

### 1.5 Retrieval diagnostics

For a typical query the top-20 retrieved cases are **20/20 the same role**, spanning a similarity range of **1.853–1.917 out of a maximum 2.0** — a 0.064 spread. Every candidate within a role looks nearly identical to the retriever, so which 10 exemplars reach the LLM is close to arbitrary. This predicts that exemplar-count and stratification-ratio ablations will move nothing.

### 1.6 Controls: does retrieval contribute anything? — IN PROGRESS

Prompt held fixed (v0), evidence varied, n=300, temperature 0:

| Condition | Evidence | Result |
|---|---|---|
| `c_retrieved` | top-20 by similarity, stratified 5/5 | pending |
| `c_random_role` | 10 random cases, same role | pending |
| `c_random_corpus` | 10 random cases, any role | pending |
| `c_zeroshot` | none — CV + JD only | pending |

Verified that the conditions genuinely differ: retrieved spans similarity 1.726–1.891 across 1 role; random-same-role 1.512–1.647 across 1 role; random-any-role 1.165–1.347 across 8 roles; zero case overlap between conditions. Random exemplars carry their real cosine similarities so the prompt structure is identical.

How to read it:

- `c_zeroshot` ≈ `c_retrieved` → the FAISS index, two-view embedding and stratified retrieval contribute nothing measurable.
- `c_random_corpus` ≈ `c_retrieved` → exemplars help as *format*; which ones you pick is irrelevant.
- `c_random_role` ≈ `c_retrieved` → role matching is the whole effect; similarity ranking within a role adds nothing.
- `c_retrieved` clearly ahead → retrieval genuinely works.

Given §1.5, the third outcome is the most likely. Any of them is a publishable result and determines what the write-up is about.

---

## 2. The evaluation framework — four layers

Report all four. Layer 1 is label-free; Layers 2–3 are the answer to "something other than accuracy"; Layer 4 keeps accuracy but in context.

### Layer 1 — Retrieval quality (no LLM)

| Metric | What it tells you |
|---|---|
| Precision@k, Recall@k, NDCG@10, MRR | Standard IR quality |
| Same-role retrieval rate | Are exemplars even the right job family? (currently 100%) |
| Exemplar diversity (mean pairwise cosine) | Detects near-duplicate evidence |
| Skill-overlap of retrieved cases | Relevance proxy that does not use the label |

### Layer 2 — LLM output quality

| # | Metric | How | Status |
|---|---|---|---|
| 2.1 | **Faithfulness** | % of `key_strengths`/`key_gaps` claims verifiable in the CV | not built |
| 2.2 | **Self-consistency** | n=5 at temperature > 0, modal decision + agreement rate | not built |
| 2.3 | **Counterfactual sensitivity** ⭐ | inject/remove a required skill → does the decision move the right way? Objective correctness, immune to label noise | not built |
| 2.4 | **Order robustness** | shuffle exemplars, measure flip rate | not built |
| 2.5 | **Calibration** | accuracy per confidence bucket, ECE, abstention | **measured — flat, §1.4** |
| 2.6 | **Format validity & latency** | % parseable JSON, s/case | **measured — 0 failures, ~12s/case** |
| 2.7 | **Reasoning-trace faithfulness** | with thinking mode on, does the answer follow the trace? | not built (Amol's Qwen setup) |
| 2.8 | **Rule adherence** | does the model follow an explicit stated procedure? | **measured — violated on 30% of cases, §1.3** |

### Layer 3 — Fairness

Counterfactual machinery from 2.3 gives this nearly free, and the dataset has a `Name` column.

- **Name-swap matched pairs**: hold the CV constant, swap for gendered / ethnically-marked name variants. Any decision change is pure bias.
- Report **flip rate per group**, **selection-rate parity**, the **80% rule**, and whether the *reasoning text* changes tone across variants.
- Run on every approach — "which retrieval method is least biased?" is a genuine contribution.

### Layer 4 — Decision quality, contextualised

Always against four reference lines:

| Reference line | Value |
|---|---|
| Random / majority | 50.0% |
| Keyword TF-IDF | 47.7% (AUC 0.474) |
| Supervised ceiling | 53.7% held-out / 58.2% 5-fold CV |
| Human agreement | **to measure — tooling ready** |

**The gold set is now the highest-value item in the project.** Every automated reference has collapsed into a 7-point band, so human agreement is the only measurement left that distinguishes a good system from a lucky one.

> **Status: tooling built and verified** (commit `8f1982d`). `data/gold/label_sheet_{sadia,amol}.html` — 150 cases drawn from the same seed-42 n=300 eval sample, so every labelled case already has a system prediction. Blind (role, JD, CV only; verified no decision field or reason string reaches the page). Different presentation order per annotator. 10 hidden repeat probes measure each annotator's own noise floor. `src/gold_set_score.py` reports self-consistency, inter-annotator kappa, humans-vs-dataset, and every system scored against both the dataset and the human consensus.

---

## 3. Approaches, ranked

Done: KG + weighted Jaccard (Amol) · MiniLM + ChromaDB + MMR (Amol) · resume vs. resume+interview (Amol, unverified) · contrastive fine-tuning of MiniLM (Amol, stops at the training cell) · two-view mpnet + FAISS + llama3.1:8b (Sadia, **now evaluated at n=300**) · keyword + supervised baselines (**done**) · prompt variants (**done, n=40**) · retrieval controls (**running**).

| Priority | Approach | Why | Status |
|---|---|---|---|
| **P0** | Keyword + supervised baselines | context for every other number | **done, §1.1** |
| **P0** | Run Step 6 with the leak removed | first real number | **done — 55.3%** |
| **P0** | Pin temperature | nothing was reproducible | **done** |
| **P0** | Retrieval controls (zero-shot / random) | does RAG contribute at all? | **running, §1.6** |
| **P1** | **Gold set labelling** | the only trustworthy correctness signal left | **tooling ready — needs 2–3 h from each annotator** |
| **P1** | **Counterfactual + name-swap harness** | objective correctness without labels, plus the fairness deliverable | next to build |
| **P1** | Two-view vs single-view ablation | Sadia's key design choice, never validated. Test at Layer 1 only — fast, and the noisy label can't obscure it | next |
| **P2** | Requirement-decomposition pipeline | turns one unverifiable judgement into ~8 checkable ones; generates rejection feedback for free. Re-run v0 vs v3-rule at n=300 with temperature 0 before investing | partially tested |
| **P2** | Self-consistency + abstention | needs §1.4 revisited once prompts change | not started |
| **P2** | Rejection feedback generator | cheapest remaining proposal deliverable; builds on `key_gaps` | not started |
| **P3** | Controlled sanity benchmark | ~500 cases where the label *is* a known function of skill coverage. Any working pipeline should score >90%. Separates "our system is broken" from "the labels are noise" | not started |
| **P3** | Cross-encoder reranking · hybrid BM25+dense · finish the fine-tune | Amol's track. **Deprioritised**: with a mid-50s ceiling and a near-random within-role ranking, better retrieval has nothing to bite on — and §1.6 may show retrieval contributes nothing at all | Amol |
| **P3** | LLM-as-judge, validated against the gold set | rubric-scored reasoning quality; worthless until validated | not started |
| **P4** | Explainability (SHAP/LIME) | proposal deliverable, complex, low information given the above | not started |

---

## 4. Split of work

**Sadia** — evaluation harness and the LLM/decision track: controls, counterfactual + name-swap probes, two-view ablation, requirement decomposition, rejection feedback.

**Amol** — retrieval track at Layer 1: finish the fine-tune, settle the transcript question on ≥300 cases (the +10pp is from 10 candidates with no saved notebook outputs and a truncated summary JSON), cross-encoder and hybrid retrieval *if* §1.6 shows retrieval matters.

**Both** — the 150-case gold set. Independently, blind, before anything else. One evening each.

**Shared infrastructure:** one results schema, one frozen test split (seed 42), one metrics module both repos import. Results now live in `data/processed/eval_*.json` and are tracked in git.

---

## 5. Open actions

1. **Label the gold set.** Both annotators. Everything in Layer 4 is blocked on it.
2. Pin the exact Qwen model ID Amol used — "qwen 3.5B" is ambiguous, and reproducibility requires the exact Ollama tag.
3. Re-run v0 vs v3-rule at n=300 with temperature 0 — the only prompt comparison worth the compute.
4. Re-export `rag-comparison-resume-vs-interview.ipynb` with outputs saved; fix the truncated `rag_comparison_summary.json`.
5. Raise with the supervisor: this dataset cannot support a "which retrieval approach wins" leaderboard — the label is roughly unlearnable and every method lands in the same 5-point band. The project is stronger as an evaluation-methodology and fairness study. Consider sourcing a second dataset with real outcomes for validation.

---

## Appendix — reproducing §0

TF-IDF (50k features, 1–2 grams, min_df=2, sublinear) + LogisticRegression (max_iter=2000), StratifiedKFold(5, shuffle, seed 0), on `data/raw/dataset.csv` (10,174 rows). Groundedness is a case-insensitive substring test of the skill term against the resume, comparing the rejected-for-that-reason group against the rest of the corpus. Held-out variants of the same models are in `src/baselines.py`.
