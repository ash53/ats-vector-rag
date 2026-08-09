# Evaluation Strategy & Next Approaches

*Written 2026-08-08. Responds to Amol's question: "accuracy ain't good — there should be something other than accuracy to test outputs of an LLM."*

---

## 0. The finding that reframes everything

Amol is right that accuracy is the wrong metric, but for a bigger reason than "the LLM is imperfect". I measured the dataset directly (scripts reproducible, ~2 min runtime).

### 0.1 The learnable ceiling is ~58–61%, not 100%

A fully supervised TF-IDF + logistic regression classifier, trained on **all 10,174 rows** with 5-fold cross-validation — i.e. the strongest thing you can do with these features, with direct access to the labels, no LLM involved:

| Input field | Accuracy | ROC-AUC |
|---|---|---|
| Resume only | **58.2%** | 0.639 |
| Resume + Job Description | 57.3% | 0.631 |
| Job Description only | 54.7% | 0.592 |
| Role only | 49.8% | 0.504 |
| Transcript only | **61.2%** | 0.693 |
| Reason_for_decision | 91.9% | 0.987 |

Class balance is 50/50 (5,060 select / 5,114 reject), so random = 50%.

**Read this carefully:** a supervised model that has *seen the labels* extracts only 8 points of signal from the resume. Amol's GraphRAG at 40% and any RAG number in the 50s are not being compared against 100% — they are competing against a ceiling of roughly 60%. The gap between "40%" and "a good system" is at most ~20 points, and most of that band is noise.

The `Reason_for_decision` row at 91.9% is a **label leak**, not a result — the reason text states the outcome. It must never enter any embedding, prompt, or index. **Sadia's Step 3 currently embeds "decision rationale" in the case-level view — this is leakage and must be removed before any number from that index is reported.**

### 0.2 The decision rationale is not grounded in the candidate's data

The top 10 reason strings cover 69% of the dataset (7,000 rows) — a clean 10-class problem. Predicting *which* reason was given:

| Input | Reason-class accuracy | Majority baseline |
|---|---|---|
| Resume | 10.5% | 10.4% |
| Transcript | 10.4% | 10.4% |
| Resume + Transcript | 10.3% | 10.4% |

**Zero signal.** The reason is assigned independently of what the candidate actually wrote.

Groundedness check — for candidates rejected *specifically* for lacking a skill, does their resume mention that skill?

| Rejection reason | Skill appears in resume | Corpus base rate |
|---|---|---|
| "Lacks hands-on experience with cloud platforms" | 29.4% | 33.1% |
| "Lacked leadership skills for a senior position" | 9.8% | 10.6% |
| "Insufficient system design expertise" | 2.3% | 3.4% |
| "No experience in back-end development" | 1.4% | 2.2% |

Candidates rejected for lacking cloud experience mention cloud at *the same rate as everyone else*. The labels are synthetic and largely uncorrelated with the content they claim to describe.

### 0.3 The job descriptions carry almost no requirements

- 3,446 unique JDs, mean length **53 words** (~370 chars)
- Only **21.7%** contain any explicit requirement phrase
- Typical JD: *"We're hiring a E-commerce Specialist to develop and deliver high-quality solutions to transform our healthcare."*

The JD is essentially the role title plus marketing copy. This is why "CV–JD semantic matching" underperforms: **there is very little in the JD to match against.** It also explains why Amol's fine-tuning showed strong retrieval@10 (0.933) but weak accuracy@1 (0.467) — the model can find the right *role cluster*, but the JD contains nothing that separates candidates within it.

### 0.4 What this means for the project

Stop optimising decision accuracy against this label. It is a ~60% ceiling over a noisy, ungrounded target, and every extra point is likely overfitting to generation artifacts.

The project is still fully deliverable — but the research question shifts from *"which retrieval approach scores highest?"* to:

> **"How do you evaluate an LLM hiring system when the ground-truth labels are unreliable — and which retrieval approach produces the most faithful, stable, and fair decisions?"**

That is a better dissertation than a leaderboard chase, it is honest, and it directly answers Amol's question. Every metric below is measurable **without trusting the dataset label**.

---

## 1. The evaluation framework — four layers

Report all four. Layer 1 is label-free. Layers 2 and 3 are the answer to "something other than accuracy". Layer 4 keeps the accuracy number but puts it in context.

### Layer 1 — Retrieval quality (no LLM involved)

Evaluate the retriever on its own, before the LLM confounds it. Cheap, deterministic, and diagnoses whether a bad decision is a retrieval failure or a reasoning failure.

| Metric | What it tells you |
|---|---|
| Precision@k, Recall@k, NDCG@10, MRR | Standard IR quality — Amol's fine-tuning notebook already produces these |
| **Same-role retrieval rate** | Are retrieved exemplars even for the same job family? (Amol has the analogous same-decision rate: 55.2% baseline) |
| **Exemplar diversity** (mean pairwise cosine of retrieved set) | Detects near-duplicate evidence — the case for MMR |
| **Skill-overlap of retrieved cases** | Grounded proxy for "is this exemplar actually relevant?" that does not use the label |

### Layer 2 — LLM output quality (the core of the new work)

These need no ground truth. This is the layer that makes the project defensible.

| # | Metric | How to measure | Why it matters |
|---|---|---|---|
| 2.1 | **Faithfulness / groundedness** | Every claim in `key_strengths` / `key_gaps` must be checkable against the CV text. Score = % of cited claims verifiable in the source. Automate with string/entity matching first, then an LLM judge. | Catches hallucinated skills — the #1 failure mode in production ATS, and a legal risk |
| 2.2 | **Self-consistency** | Run each case n=5 times at temperature > 0. Report modal decision + agreement rate. | Measures stability. A system that flips on 40% of cases is unusable regardless of accuracy |
| 2.3 | **Counterfactual sensitivity** ⭐ | Inject a required skill into a CV → does the decision move toward select? Delete it → toward reject? Report *directional correctness rate*. | **This is the key unlock.** An objective correctness signal that is completely immune to label noise. You are testing whether the model responds to evidence, not whether it guesses the synthetic label |
| 2.4 | **Position / order robustness** | Shuffle the order of the 10 exemplars, re-run. Flip rate = position bias. | Known LLM weakness; trivially measurable; a real finding for the report |
| 2.5 | **Calibration** | Bucket by the model's own `confidence` field → accuracy per bucket; report ECE. Then test abstention: "flag low-confidence for human review". | Turns the 60% ceiling into a usable product: auto-decide the confident 40%, escalate the rest |
| 2.6 | **Format validity & latency** | % parseable JSON, mean seconds/candidate, token cost | Engineering baseline; Sadia's Step 5 needs this anyway |
| 2.7 | **Reasoning-trace faithfulness** | With Qwen's thinking mode on: does the final answer follow the trace? Sample cases where the trace argues reject and the output says select. | Novel, cheap, and directly exploits the thinking-mode setup already in use |

### Layer 3 — Fairness (a proposal deliverable, and now easy)

Counterfactual machinery from 2.3 gives this almost for free — and the dataset has a `Name` column, which is perfect for it.

- **Name-swap test**: hold the CV constant, swap the name for gendered / ethnically-marked variants (matched-pair audit design, as in Bertrand & Mullainathan). Any decision change is pure bias — the qualifications are identical.
- Report **flip rate per demographic group**, plus **selection-rate parity** and the **80% (four-fifths) rule**.
- Also test: does the *reasoning text* change tone or content across name variants?
- Run the same probe on every approach — "which retrieval method is least biased?" is a genuine contribution.

### Layer 4 — Decision quality, correctly contextualised

Keep accuracy/precision/recall/F1, but **always report them against three reference lines**:

| Reference line | Value | Status |
|---|---|---|
| Random / majority | 50% | — |
| Keyword baseline (TF-IDF cosine + threshold) | to measure | required by proposal |
| **Supervised ceiling** (TF-IDF + LogReg, 5-fold CV) | **58.2%** | measured, see §0.1 |
| Human agreement on a gold subset | to measure | see below |

**Build the human gold set.** 150 cases, both of you label independently, blind, before seeing the dataset label. This gives you (a) inter-annotator agreement — the real human ceiling, (b) a trustworthy test set, (c) a measurement of how often the *dataset itself* is wrong. If you two agree with each other 80% of the time and with the dataset 55% of the time, you have proven the label is the problem. That single result justifies the whole re-framing.

---

## 2. Approaches not yet tried, ranked

Already done: KG + weighted Jaccard (Amol), MiniLM + ChromaDB + MMR (Amol), resume vs. resume+interview (Amol, unverified), contrastive fine-tuning of MiniLM (Amol, incomplete — stops at the training cell), two-view mpnet + FAISS + llama3.1:8b (Sadia, never evaluated).

| Priority | Approach | Why it's next | Effort |
|---|---|---|---|
| **P0** | **Supervised + keyword baselines** | Already measured for you (§0.1). Wrap as `baseline_keyword.py` + `baseline_supervised.py` so every later number has context. Nothing else is interpretable without them. | 2h |
| **P0** | **Run Sadia's Step 6 once** — after removing the rationale leak | Three months of pipeline work with zero numbers. Unblocked since the FAISS index finished on 2026-05-12. | 1 day |
| **P1** | **Requirement-decomposition (structured/agentic) pipeline** ⭐ | Instead of one-shot decide: (1) LLM extracts explicit requirements from the JD, (2) for each requirement, retrieve CV evidence and score met / partial / missing with a quote, (3) aggregate to a rubric score, (4) threshold. **Turns one unverifiable judgement into ~8 individually checkable ones** — so it directly serves the Layer-2 metrics and produces auditable output. Also generates the rejection-feedback content for free. Highest expected payoff of anything on this list. | 3–4 days |
| **P1** | **Cross-encoder re-ranking** | Retrieve top-50 bi-encoder → rerank with `ms-marco-MiniLM-L-6-v2`. The single biggest standard retrieval win and nobody has tried it. Measured purely at Layer 1, so the noisy label can't obscure the result. | 1 day |
| **P1** | **Hybrid retrieval: BM25 + dense, fused with RRF** | Lexical and semantic signals are complementary; also merges Sadia's vector work with Amol's entity-overlap idea without a full graph rebuild. | 1–2 days |
| **P2** | **Self-consistency + abstention** | n=5 sampling, majority vote, escalate low-agreement cases. Likely the largest *real* accuracy gain available, and it produces the calibration story (2.5). | 1 day |
| **P2** | **Finish the fine-tuned bi-encoder** | Amol is 20 cells from a before/after comparison. Baseline is already logged (accuracy@1 0.467, NDCG@10 0.438, MRR@10 0.626) — just needs Part 6 onward to run. Evaluate at Layer 1 only. | 1 day |
| **P2** | **Resolve the transcript question properly** | Amol's +10pp (30%→40%) is from 10 candidates with no saved notebook outputs and a truncated summary JSON — not yet evidence. Transcripts are the strongest single field (61.2% supervised vs 58.2% for resume), so it's worth settling on ≥300 cases. Note it is arguably out of scope: a *screening* ATS sees no interview. Decide as a group and document it. | 1 day |
| **P3** | **Controlled sanity benchmark** | Generate ~500 cases where the label *is* a known function of skill coverage (e.g. select iff ≥70% of required skills present). Any competent pipeline should score >90% here. If yours doesn't, the bug is in the pipeline, not the data. Cleanly separates "our system is broken" from "the labels are noise" — and that separation is the backbone of the results chapter. | 1 day |
| **P3** | **LLM-as-judge, validated** | Rubric-scored reasoning quality (1–5: relevance, groundedness, specificity, tone). Must be validated against the human gold set before it is trusted — report judge–human correlation, or it is worthless. | 2 days |
| **P4** | Exemplar-count and single-vs-two-view ablations | Already on Sadia's roadmap. Cheap, but only meaningful once Layers 1–2 exist. | 1 day |

---

## 3. Suggested split

**Amol** — retrieval track (Layer 1): cross-encoder reranking, hybrid BM25+dense, finish the fine-tune, settle the transcript question on a larger sample. One comparison table, one metric set, all approaches.

**Sadia** — evaluation harness + LLM track (Layers 2–4): remove the rationale leak, run Step 6, build the counterfactual + name-swap + self-consistency probes as a reusable module, then the requirement-decomposition pipeline.

**Both** — the 150-case human gold set. Do this first and independently; it is the foundation of Layer 4 and takes an evening each.

**Shared infrastructure, agree before building:** one `results.json` schema, one fixed test split (seed pinned, 300+ cases), one metrics module both repos import. Right now nothing is comparable across the two repos because the samples, models, and metrics all differ.

---

## 4. Immediate actions

1. Pin the exact model IDs — "qwen 3.5B" is ambiguous (Qwen2.5-3B? Qwen3-4B?). Reproducibility requires the exact Ollama tag, and it goes in the report.
2. Remove `Reason_for_decision` from Sadia's Step 3 case-level embedding (leak, §0.1).
3. Commit and push the untracked work in both repos — `MEETING_NOTES.md`, `ROADMAP.md`, `step5`, `step6`, `requirements.txt` on Sadia's side; `PROJECT_OVERVIEW.md` on Amol's.
4. Re-export `rag-comparison-resume-vs-interview.ipynb` with outputs saved, and fix the truncated `rag_comparison_summary.json`.
5. Freeze the shared test split and the metrics module before anyone runs another experiment.

---

## Appendix — reproducing §0

Scripts used: TF-IDF (50k features, 1–2 grams, min_df=2, sublinear) + LogisticRegression (max_iter=2000), StratifiedKFold(5, shuffle, seed 0), on `data/raw/dataset.csv` (10,174 rows). Groundedness check is a case-insensitive substring test of the skill term against the resume, comparing the rejected-for-that-reason group against the rest of the corpus.
