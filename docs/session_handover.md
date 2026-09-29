# Session handover

*Written 2026-09-29. Covers the work from 2026-08-08 onward.*

Read `RESULTS.md` for every number, `EVALUATION_AND_APPROACH_PLAN.md` for the
reasoning, `docs/results_chapter_skeleton.md` for how it becomes a write-up.

---

## READ THIS FIRST — the conclusion changed

The four experiments that finished while the session was idle include one that
**overturns the previous headline**, and the older documents have not caught up.

Until now the story was: *the dataset's labels are unlearnable, everything scores
in the mid-50s, so the ceiling belongs to the data, not to our system.*

The sanity benchmark disproves the second half of that.

### The sanity benchmark result

500 generated cases where the answer is knowable by construction: the job
description lists its required skills explicitly, the CV either evidences them
or does not, and **select ⟺ the CV covers ≥70% of them**. Zero label noise.

| System | Accuracy |
|---|---|
| Oracle (counts skill overlap) | 100.0% |
| Bag of words (TF-IDF + logistic regression) | 78.0% |
| **Our pipeline (LLM zero-shot)** | **51.0%** |

Broken down by how many of the 5 required skills the CV actually evidenced:

| Skills evidenced | Correct answer | Our accuracy | We said "select" |
|---|---|---|---|
| 0 / 5 | reject | 10% | 90% |
| 1 / 5 | reject | 0% | 100% |
| 2 / 5 | reject | 0% | 100% |
| 3 / 5 | reject | 0% | 100% |
| 4 / 5 | select | 100% | 100% |
| 5 / 5 | select | 100% | 100% |

**The system said "select" on 99% of cases.** It is right on every case whose
answer is "select" and wrong on almost every case whose answer is "reject",
because it is not deciding — it is defaulting.

This is not a dataset problem. The requirements were listed. The evidence was
unambiguous. There was no label noise. A bag-of-words model managed 78%. Our
pipeline scored 51%, which on a balanced set is what you get for answering
"yes" every time.

### What the revised conclusion is

**Both** things are true, and the project needs to say both:

1. **The dataset cannot support the task.** Supervised ceiling 58.2%; stated
   reasons uncorrelated with the CVs; job descriptions averaging 53 words.
2. **The decision layer is also defective.** It cannot perform requirement
   matching even when the task is trivial and well-posed.

The second is the stronger and more specific finding, and three independent
experiments now converge on the same mechanism:

- **Counterfactual** — inject a required skill the CV lacks: the decision
  changed in **10 of 100** cases. Remove one it has: **3 of 100**. It barely
  responds to the specific evidence that should decide the case.
- **Input ablations** — it responds strongly to whether a CV is *present*
  (select rate 87% with a real CV, 5.7% with none) but weakly to what the CV
  *says* (88.7% even when truncated to 200 characters).
- **Sanity benchmark** — 99% select regardless of whether the candidate covers
  0 or 5 of the required skills.

**The model's effective decision rule is "is there a plausible-looking CV
here?", not "does this candidate meet the requirements?"** That sentence is the
project's main finding, and it is now backed by three separate measurements.

---

## Documents that are now WRONG and must be fixed

I did not update these before the session ended. Do this first.

| File | What is stale |
|---|---|
| `docs/meeting_brief.md` | Says the ceiling belongs to the dataset and the pipeline is sound. The sanity result contradicts this. **Do not present it as written.** |
| `EVALUATION_AND_APPROACH_PLAN.md` | §0.4 and §1.6 frame the null as a data property. Needs the revised two-part conclusion, and §1.6's "how to read it" for the sanity benchmark is now answered. |
| `progress_overview.md` | Same framing issue. |
| `docs/results_chapter_skeleton.md` | §5.13 says "write this only after the number is in". The number is in, and it points the other way. |

`RESULTS.md` and `figures/` were regenerated and **are** current.

---

## Correction: a file I said existed does not

In the last session I told you I had written `docs/methods_explainer.md` with
definitions of every method and statistic. **I never created it.** The chat
message contained the definitions; no file was written. The content of that
message is not saved anywhere in the repo. If you want it as a document it has
to be written from scratch.

---

## Results you have not seen

### Input ablations, n=300 (zero-shot; compare to 55.7% / 87.0% with the real CV)

| Condition | Accuracy | Select rate |
|---|---|---|
| Someone else's CV | 48.7% | 33.3% |
| No CV at all | 49.0% | 5.7% |
| CV truncated to 200 chars | **52.7%** | **88.7%** |

The truncated row is the informative one: cut the CV to two lines and behaviour
barely changes from the full CV. Presence matters; content mostly does not.

### Counterfactual sensitivity, n=100 per arm

| Arm | Unchanged | Moved as expected | Moved against the evidence |
|---|---|---|---|
| Inject a required skill the CV lacks | 90 | 10 | 0 |
| Remove a required skill the CV has | 97 | 3 | 0 |

Nothing moved the *wrong* way, which is worth stating — but almost nothing
moved at all.

### Fairness, n=150 per group

| Group | Select rate |
|---|---|
| anglo_female | 90.7% |
| black_female | 90.0% |
| anglo_male | 89.3% |
| black_male | 88.0% |

Largest gap 2.7 points; impact ratio **0.97**, which **passes** the four-fifths
rule. **But 7 of 150 decisions (4.7%) changed when only the name changed** on an
otherwise byte-identical CV.

Report both halves honestly: no systematic group-level disparity was detected at
this sample size, yet individual decisions are not stable under a change that
must not matter. Also note the ceiling effect — at an 88–91% select rate there
is little room for a disparity to appear. A fairness test on a system that
approves nearly everyone is weak by construction, and that limitation belongs in
the write-up.

---

## Repository state

On `main`, in sync with origin, everything through `e458fad` pushed.

**Uncommitted** (new results + the meeting brief + regenerated tables/figures):

```
 M data/processed/eval_ablations_n300.json
 M RESULTS.md · figures/*
 ?? data/processed/eval_counterfactual_n100.json
 ?? data/processed/eval_fairness_n150.json
 ?? data/processed/eval_sanity.json
 ?? docs/meeting_brief.md · docs/session_handover.md
 M src/make_results_table.py   (added the sanity-benchmark section)
```

These were held back deliberately — you asked for one batched commit at the end.

---

## What to do next, in order

1. **Commit the above.** One commit, message should lead with the sanity result
   since it changes the conclusion.
2. **Fix the four stale documents** listed above before the meeting. The meeting
   brief in particular would have you presenting a claim your own data now
   contradicts.
3. **Check whether the defect is the model or the prompt.** The cheapest
   decisive experiment left: re-run the sanity benchmark with a bigger model, or
   with the v3 requirement-checklist prompt instead of the free-form one. If a
   checklist prompt scores 90% on the sanity benchmark, the defect is the prompt
   and it's fixable. If it still scores 51%, it's the model. Either answer is
   publishable and it takes about an hour.
4. **Label the gold set** — still unstarted, still needs both annotators, still
   the only measurement that compares the system to human judgement.
5. **Then write up.**

---

## Things worth knowing that are not obvious from the code

- **Runs are reproducible now** (`step5.TEMPERATURE = 0`, fixed seed) but were
  not before 2026-08-09. Any number in a doc dated earlier was sampled at
  temperature 0.8, where the same prompt on the same 40 cases scored 45.0% then
  50.0%.
- **Three FAISS indexes exist on purpose.** `faiss_index.bin` is live (768-d,
  case view only). `faiss_index_twoview.bin` reproduces everything committed
  before the skill view was dropped. `faiss_index_LEAKY.bin` is the pre-fix
  index that embedded `decision_reason` — kept so the leak's effect stays
  measurable. None are in git; they are regenerated or copied.
- **`RESULTS.md` and `figures/` are generated.** Never hand-edit them; run
  `src/make_results_table.py` and `src/make_figures.py`.
- **`exp_retrieval_views.py` reads the two-view index by design** and refuses
  any index that is not 1536-d, because splitting the live 768-d index in half
  would silently produce two meaningless 384-d halves.
- **One experiment never run:** self-consistency (n=5 sampling at temperature >
  0). The harness supports it — `call_llm` takes `temperature` and `seed`.
