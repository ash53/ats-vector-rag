# Results — chapter skeleton

**This is scaffolding, not a draft.** It fixes the order of the argument, states
the facts that are already established, and marks every place where the writing
has to be yours with `[WRITE]`. Numbers come from `RESULTS.md`, which is
generated from the raw result files — quote them from there rather than from
this file, so a re-run cannot leave a stale figure behind.

Status key: **✓ measured** · **⧗ running** · **○ not started**

---

## The argument in one paragraph

The chapter has to earn one claim: *on this dataset the hiring decision is
largely not learnable from the candidate's data, every method we built performs
within a few points of every other, retrieval-augmented generation adds nothing
over zero-shot prompting, and the interesting question therefore becomes how you
evaluate such a system at all.* Everything below is ordered to build that claim
in the order a sceptical reader would demand it — first show the task is
degenerate, then show your system is not at fault, then show what can still be
measured meaningfully.

---

## 5.1 Experimental setup ✓

`[WRITE]` One paragraph fixing the protocol so every later number is
interpretable without re-explanation.

Established facts to state:

- Dataset: Kaggle AI Recruitment Pipeline, 10,174 records, 45 roles, balanced
  (5,060 select / 5,114 reject), so random and majority-class both score 50%.
- Test protocol: stratified sample, seed 42, n=300 unless stated. Identical
  cases across every condition, so all comparisons are **paired**.
- Model: `llama3.1:8b` via Ollama, greedy decoding (temperature 0, fixed seed).
- Embeddings: `all-mpnet-base-v2`, FAISS `IndexFlatIP`.
- Significance: binomial CIs for single proportions; **McNemar** for paired
  system comparisons; paired *t*-tests for per-query retrieval metrics.

`[WRITE]` State plainly that the temperature was **not** pinned for the earliest
runs and that this was found by re-running an identical configuration and
getting 45.0% then 50.0%. Reporting this is not an admission of sloppiness; it
quantifies the noise floor and justifies why small-sample comparisons are not
used to rank anything.

---

## 5.2 The dataset does not support the task ✓

This section must land before any system result, or every later number gets read
against an implicit 100% ceiling.

**Table 5.1** — supervised probe by input field (`RESULTS.md` §1 and plan §0.1).

| Finding | Value | Source |
|---|---|---|
| Supervised ceiling from the resume | 58.2% acc, AUC 0.639 | 5-fold CV, all 10,174 rows |
| From the transcript | 61.2%, AUC 0.693 | " |
| From `Reason_for_decision` | 91.9%, AUC 0.987 | **label leak, excluded** |
| Reason class predictable from CV | 10.5% vs 10.4% baseline | 10-class, 7,000 rows |
| JD length / with any requirement | 53 words / 21.7% | full corpus |

`[WRITE]` Interpret the groundedness check — candidates rejected for "lacking
cloud experience" mention cloud at 29.4% against a 33.1% corpus base rate. The
stated reasons are not descriptions of the candidates. Say what that implies for
any supervised or retrieval method trained or evaluated on these labels.

`[WRITE]` Methodological note on the leak found in our own indexing (§5.7),
since the 91.9% row is what exposed it.

---

## 5.3 Decision quality against reference lines ✓

**Table 5.2** — `RESULTS.md` §1, n=300.

Facts: RAG 55.3% (CI 49.5–61.0) · supervised 53.7% · always-select 50.0%
(F1 66.7%) · keyword TF-IDF 47.7% (AUC 0.474). Paired McNemar: RAG beats
always-select p=0.037; indistinguishable from supervised p=0.75.

`[WRITE]` Two points that need care:

1. The keyword baseline — the traditional ATS the proposal set out to beat — is
   **at chance**. Frame this as a property of 53-word job descriptions, not a
   victory.
2. F1 is actively misleading here: always-select scores 66.7% F1 by doing
   nothing. Argue for reporting the select rate beside every F1. Our system's
   82.7% select rate is the number that explains its 88% recall.

---

## 5.4 Retrieval contributes nothing ✓

The chapter's strongest result. Prompt fixed, evidence varied, same 300 cases.

**Table 5.3** — `RESULTS.md` §2.

Facts: retrieved 54.3% · random same-role 55.0% · random any-role 55.0% ·
**zero-shot 55.7%**. Every paired McNemar between conditions p ≥ 0.50.

`[WRITE]` Establish that this is a real null, not a broken experiment. The
evidence to cite: conditions share zero exemplar cases and span clearly
different similarity ranges; predictions differ on 10–22 of 300 between
conditions; generated reasoning differs on essentially every case. The model
reads the exemplars — they do not change its decision.

`[WRITE]` State the limitation honestly: with a ~55–58% ceiling there is little
headroom, so the experiment has limited power to detect a *small* genuine
benefit. What it establishes is that no condition beats any other and that
zero-shot is not worse.

---

## 5.5 Why retrieval cannot help — the mechanism ✓

`[WRITE]` §5.4 observes the null; this explains it, which is what turns a
negative result into a contribution.

**Table 5.4** — `RESULTS.md` §5, 300 queries leave-one-out, k=10.

Facts: exemplars are only useful if their outcomes correlate with the query's.
Best view reaches **56.6% same-decision@10** — above the 50% no-information line
(p=6×10⁻⁷) but carrying just 6.6 points of signal before the LLM sees anything.
Retrieved exemplars are near-duplicates (mean pairwise cosine 0.93–0.99) and the
index barely separates its own neighbours (similarity spread 0.021–0.038).

`[WRITE]` Draw the consequence: cross-encoder reranking, hybrid retrieval and
fine-tuned embeddings are all competing for those 6.6 points. This is why we did
not pursue them, and it is a prediction other work on this dataset can test.

---

## 5.6 Does the system read the candidate at all? ⧗

**Table 5.5** — `RESULTS.md` §3, zero-shot throughout, compare to `c_zeroshot`
(55.7% accuracy, 87.0% select rate).

Facts so far: swapped CV — a different candidate's CV against this JD and label
— gives **48.7% accuracy, 33.3% select rate**. Empty-CV and truncated-CV
conditions still running.

`[WRITE]` The interpretation is genuinely interesting and needs stating
carefully: the select rate moves 87% → 33% when the CV is swapped, so the system
is **highly sensitive** to whether a CV plausibly matches a role. It is not a
constant function. Yet that sensitivity converts to only 55.7% accuracy against
the real labels. The model discriminates; the labels do not reward it. This is
the cleanest evidence that the ceiling belongs to the dataset rather than the
pipeline.

---

## 5.7 Two defects found in our own pipeline ✓

`[WRITE]` A short, unembarrassed section. Both were found by measurement rather
than inspection, which is itself an argument for the evaluation framework.

1. **Label leak.** Step 3 embedded `decision_reason` into every indexed vector.
   That field states the outcome; a classifier recovers the label from it alone
   at 91.9%. It was also an asymmetry bug — the query side never included it, so
   indexed and query vectors came from different templates.
2. **Unjustified design choice.** The two-view embedding (case ‖ skill) was this
   pipeline's distinctive contribution. Measured, the concatenation is
   indistinguishable from the case view alone (−0.6pp, p=0.50) at double the
   dimension; the skill view alone is significantly worse (p=0.007). Removed;
   index halved from 62 MB to 31 MB with no loss.

---

## 5.8 Prompt sensitivity, calibration and rule adherence ✓

**Table 5.6** — `RESULTS.md` §4. Carry the small-sample warning: n=40, before
temperature was pinned, so the *ranking* is not trustworthy.

Facts that survive replication:

- Select rate is movable (90% → 47.5%) with **no accuracy gain** — false
  positives trade for false negatives roughly one-for-one, the signature of a
  near-random underlying ranking.
- Confidence is uninformative at n=300: "high" on 276 cases at 56% accuracy,
  "medium" on 24 at 50%. Confidence-based abstention has nothing to threshold on.
- The model contradicted its own stated aggregation rule on **30%** of cases;
  moving the arithmetic into code recovered 5 points.

`[WRITE]` Generalise the last point: do not ask an LLM to aggregate when the
aggregation can be code.

---

## 5.9 Counterfactual sensitivity ⧗

**Table 5.7** — `RESULTS.md` §6.

`[WRITE]` Motivate the method before the numbers: it needs no trustworthy label.
Inject a required skill the CV lacks, or remove one it has, and the decision must
not move against the evidence. Report unchanged / expected / contradictory per
arm, and the directional-correctness rate among decisions that moved (50% = coin
flip).

---

## 5.10 Fairness ⧗

**Table 5.8** — `RESULTS.md` §7.

`[WRITE]` Method first: matched-pair audit, CVs byte-identical, only the name
changed, following Bertrand & Mullainathan (2004). Any change in decision is
bias, measured rather than inferred. Report per-group select rates, the
four-fifths impact ratio, and the share of decisions that change on the name
alone.

`[WRITE]` The limitation belongs *in the text*, not a footnote: names are a
coarse and contested proxy for perceived demographic group, they confound
ethnicity with nationality and class, and this measures a model's response to a
name — not the experience of any real group.

---

## 5.11 Rejection feedback and its failure modes ⧗

**Table 5.9** — `RESULTS.md` §8: ungrounded / contradicted / neutrality failure
rates.

`[WRITE]` This is the one output that reaches a person, and the qualitative
finding is worth a full paragraph and a verbatim example. In the first
two-message batch, the system told a devops candidate they had not evidenced
containerisation, cloud platforms or scripting. Their CV lists **kubernetes,
aws, azure, gcp and python**. Three false statements about their own
application, none of them requirements the job description stated.

`[WRITE]` Connect it to §5.2: a model asked to justify a decision whose label is
uncorrelated with the candidate's data will invent plausible-sounding reasons.
Fabrication is the predictable consequence, not a surprise. Argue that automated
groundedness and contradiction checks are a *requirement* for any deployed
system of this kind, and note the deliberate distinction between an
**ungrounded** gap (unsupported) and a **contradicted** one (false).

---

## 5.12 Comparison with human judgement ○

**Table 5.10** — from `gold_set_score.py`.

`[WRITE]` Blocked until both annotators complete the 150-case gold set. Report
annotator self-consistency from the hidden repeat probes, inter-annotator
agreement and Cohen's κ, each annotator against the dataset label, and every
system against human consensus.

The prediction to state and then test: if two annotators agree with each other
substantially more than either agrees with the dataset, the dataset label is the
unreliable party — which would corroborate §5.2 from an entirely independent
direction.

---

## 5.13 Does the pipeline work when the task is well posed? ⧗

**Table 5.11** — `RESULTS.md`, sanity benchmark.

Facts: 500 generated cases, select ⟺ the CV evidences ≥70% of the required
skills. Zero label noise. Two reference lines — oracle on the skill-overlap
count **100%**, bag-of-words TF-IDF **78%**.

`[WRITE]` Explain why bag-of-words caps at 78%: the rule is *relational* (which
skills appear in **both** documents) and a bag of words cannot represent that.
That makes 78% the line the LLM must beat, because relational comparison is
exactly what an LLM is supposed to add over lexical matching.

`[WRITE]` Then whichever way it falls: >90% means the pipeline works and the
mid-50s is the data's ceiling; ≤78% means the system cannot do the task even
with the answer stated in the inputs, which would change the chapter's
conclusion. **Write this section only after the number is in.**

---

## 5.14 Threats to validity ✓

`[WRITE]` Be first to say all of these:

- **Single dataset.** Every finding is about this corpus. The RAG null may not
  generalise to a dataset with real outcomes and substantive job descriptions —
  and §5.5 gives the mechanism that would have to be absent for it to transfer.
- **Single model.** One 8B model, locally hosted. A larger or reasoning-tuned
  model might behave differently, particularly on §5.13.
- **Sample sizes.** n=300 gives roughly ±6 points; differences below that are
  not resolvable. The n=40 runs are reported but never used for ranking.
- **Ceiling effects.** With everything between 47% and 56%, the power to detect
  small genuine improvements is low. The null results are "no detectable
  difference", not "proven identical".
- **Names as a proxy** (§5.10).
- **The sanity benchmark is synthetic** (§5.13) — it shows the pipeline can do a
  well-posed version of the task, not that it would do a real one.

---

## 5.15 Summary of findings ✓

`[WRITE]` Six numbered claims, each with its evidence and section reference.
Draft ordering:

1. The labels are largely unlearnable from the candidate data (§5.2).
2. Every method scores within a 47.7–55.7% band (§5.3).
3. Retrieval adds nothing over zero-shot (§5.4), and we can say why (§5.5).
4. The system is input-sensitive but that sensitivity does not track the labels
   (§5.6).
5. Generated feedback fabricates, and the fabrications are detectable
   automatically (§5.11).
6. `[WRITE]` — fairness and human-agreement claims, once §5.10 and §5.12 land.

---

## Figures worth making

None exist yet; all are derivable from `data/processed/eval_*.json`.

1. **Forest plot** — every system's accuracy with 95% CIs against the 50% line.
   Makes the "everything is in one band" claim in a single image.
2. **Grouped bars, accuracy vs select rate** across the four control conditions.
   Shows the null and the select bias together.
3. **Select rate by CV condition** (real / swapped / truncated / empty) — the
   87% → 33% collapse is the most striking single figure available.
4. **Same-decision@10 by embedding view**, with the 50% no-information line.
5. **Calibration plot** — accuracy by stated confidence, against the diagonal.
6. **Fairness**: select rate by demographic group with the four-fifths threshold.

`[WRITE]` Load the `dataviz` guidance before building these, and keep one
consistent palette across all six.
