# TimelineCronQ-R Reconstruction

This README documents how we reconstructed the cleaned **TimelineCronQ-R** benchmark used in the SCoP experiments.

The pipeline follows the benchmark-preparation procedure described in the SCoP paper.
It leaves the underlying temporal knowledge graph unchanged and repairs only the QA annotation layer, including question text, answer and supporting-event consistency, duplicates, filtering, balancing, and the final split.
The reconstruction follows these principles:

- the underlying temporal KG is kept unchanged;
- the QA annotation layer is conservatively repaired;
- answer sets, supporting events, duplicate handling, answer formatting, targeted relabeling, and final splits are deterministic and auditable;
- `qtype` is derived deterministically from the existing `temporal_relation` to provide a clearer taxonomy of temporal reasoning patterns;
- the dataset retains the original question together with a `qa_repair_thought` trace from the LLM rewriting stage;
- stratified manual checks were used to validate the rewritten questions.

---

## 1. Required Inputs

The pipeline expects the following files in the reconstruction working directory:

```text
TimelineCronQ-R_repair_code/
├── unified_kg_cron_questions_all.json
└── TimelineCronQ.pkl
```

### 1.1 QA Input

```text
unified_kg_cron_questions_all.json
```

This union of raw QA records is the starting point for the reconstruction pipeline.
It has already received an initial answer-format normalization pass.

### 1.2 Temporal KG Graph

```text
TimelineCronQ.pkl
```

This igraph-based temporal KG is built from the original event quadruples released with TimelineTKGQA.
The deterministic gold-repair stage uses it for KG-backed semantic execution and verification.

---

## 2. Final Directory Structure

```text
TimelineCronQ-R_repair_code/
├── repair.sh
├── 1_repair_duration.py
├── 2_repair_timeline_gold.py
├── 3_repair_leakage.py
├── 4_final_format.py
├── 5_repair_question.py
├── intermediate_data/
└── final_data/
```

### File Roles

| File                          | Purpose                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------- |
| `repair.sh`                 | Main reconstruction launcher for Steps 1–5                                                 |
| `1_repair_duration.py`      | Repairs known malformed duration-question templates under deterministic validity checks     |
| `2_repair_timeline_gold.py` | Deterministic gold-answer and gold-event repair with KG-backed semantic verification        |
| `3_repair_leakage.py`       | Removes duplicate-question leakage and merges duplicated simple cases                       |
| `4_final_format.py`         | Final formatting, filtering, relabeling, balancing, qtype assignment, and 8:1:1 resplitting |
| `5_repair_question.py`      | LLM-based surface-form rewriting of questions and constrained answer-format polishing       |

---

## Reconstruction Pipeline

Run the full reconstruction pipeline with:

```bash
bash repair.sh
```

The reconstruction has five stages:

1. repair malformed duration-question templates;
2. perform deterministic gold repair and consistency filtering;
3. remove duplicate-question leakage;
4. apply final formatting, targeted relabeling, balancing, and resplitting;
5. apply constrained LLM-based surface-form rewriting of questions.

Steps 1–4 repair and restructure the dataset deterministically.
Step 5 only polishes question wording and preserves the underlying QA semantics.

---

## 4. Step-by-Step Reconstruction

### Step 1. Duration-Question Text Repair

This step:

- repairs known malformed duration-related question templates;
- validates offset consistency before rewriting a question;
- modifies only the `question` field;
- does **not** modify answers, events, labels, or temporal relations.

The fixes cover malformed duration phrasing, including missing `before` / `after` connectors and a small number of malformed subject-prefix templates.

---

### Step 2. Deterministic Gold Repair and Semantic Verification

This step:

- uses `events` as canonical event facts;
- uses `temporal_relation` as a hard temporal program;
- uses `answer_type` to determine answer projection;
- executes KG-backed semantic verification over `TimelineCronQ.pkl` when resolvable.

For each record, the script writes repair metadata including:

- `source_event_gold`;
- `semantic_gold`;
- repair status;
- program type;
- diagnostic notes.

The retained-clean policy is:

- keep `ORIGINAL_EXACT`;
- keep `ORIGINAL_SUBSET_OF_REPAIRED` when the semantic gold size is at most 10;
- keep non-empty `REPAIRED_SUBSET_OF_ORIGINAL`;
- discard cases that fail the retained-clean policy.

#### Step 2 Report Summary

| Quantity               |  Count |
| ---------------------- | -----: |
| Input QA records       | 39,216 |
| Retained clean records | 24,391 |

Semantic-execution coverage:

| Mode                                      |  Count |
| ----------------------------------------- | -----: |
| KG-backed exact semantic execution        | 33,039 |
| Source-event operator semantics preserved |  4,014 |

---

### Step 3. Duplicate-Question Cleaning

This step uses the following policy:

- duplicate key = exact stripped `question` string;
- duplicated `medium` / `complex` questions are removed;
- duplicated `simple` questions are merged into one representative sample;
- merged simple cases aggregate and deduplicate `answer` and `events`;
- merged simple samples are redistributed with an approximate 8:1:1 train/validation/test allocation;
- the stage leaves no remaining duplicate question groups.

#### Step 3 Report Summary

| Quantity                                  |  Count |
| ----------------------------------------- | -----: |
| Input records                             | 24,391 |
| Duplicate question groups                 |  1,904 |
| Records inside duplicate groups           |  4,021 |
| Final records                             | 22,272 |
| Net removed records                       |  2,119 |
| Remaining duplicate groups after cleaning |      0 |

Simple-case merging:

| Quantity                        | Count |
| ------------------------------- | ----: |
| Simple merge groups             | 1,902 |
| Merged simple input records     | 4,017 |
| Merged simple output records    | 1,902 |
| Simple records removed by merge | 2,115 |

---

### Step 4. Final Formatting, Relabeling, Balancing, and Resplitting

This stage:

1. answer normalization;
2. invalid-answer removal;
3. removal of cases with more than 10 retained answers;
4. removal of obsolete `source_kg_id` and previous split assignments;
5. deterministic event ordering;
6. answer ordering when answers can be aligned exactly to event fields;
7. targeted `complex`-to-`medium` relabeling for a specific over-labeled subtype;
8. assignment of `qtype` from `temporal_relation`;
9. balancing of `simple`, `medium`, and `complex` levels;
10. stratified 8:1:1 resplitting by:
    - `question_level`
    - `question_type`

#### New QType Assignment

All original dataset labels are preserved.
To make temporal reasoning patterns easier to analyze, the pipeline adds a `qtype` field derived deterministically from `temporal_relation`:

| `temporal_relation` pattern or value | Assigned `qtype` |
| --- | --- |
| `timeline`, `rank_start_time`, `rank_end_time` | `timeline_structure` |
| `union`, `intersection`, `sum`, `average` | `temporal_operation` |
| `duration_before`, `duration_after`, `duration_during` | `relative_temporal` |
| <code>duration_&lt;N&gt;\s+days\s+(before&#124;after)</code> | `quantitative_temporal` |
| Any `&`-separated component matches an interval-topology relation such as `x d y`, `x di y`, `x o y`, or `x oi y` | `interval_relation` |
| All `&`-separated components are pure ordering relations, namely `x < y` or `x > y` | `temporal_order` |

This mapping provides a compact taxonomy for analyzing model performance across temporal reasoning patterns in SCoP.

#### Targeted Complex-to-Medium Relabeling

A predefined deterministic rule relabels one narrow metadata-defined subtype.
A record changes from `complex` to `medium` only when all of the following conditions hold:

```text
question_level    == "complex"
question_type     == "timeline_position_retrieval*3"
answer_type       == "relation_union_or_intersection"
temporal_relation == "union"
```

This rule covers a template family whose original metadata can overstate its effective reasoning complexity.
Although the stored record may have a three-event template label, the question and gold answer concern the union of two explicitly queried event intervals.

A representative example is:

```json
{
  "question": "What are the combined time periods for Les Roberts member of sports team Brentford F.C. and Mike Thompson position held United States representative?",
  "answer": "1930-01-01 - 1931-01-01; 2013-01-01 - 2013-01-01",
  "events": [
    "Les Roberts|member of sports team|Brentford F.C.|1930-01-01|1931-01-01",
    "Mike Thompson|position held|United States representative|2013-01-01|2013-01-01",
  ],
  "question_level": "complex",
  "question_type": "timeline_position_retrieval*3",
  "answer_type": "relation_union_or_intersection",
  "temporal_relation": "union"
}
```

The question and answer depend on the union of the two explicitly mentioned event intervals.
These records are therefore relabeled as `medium` before balancing and resplitting.

Outputs:

```text
final_data/
├── step3_leakage_cleaned.final_all.json
├── step3_leakage_cleaned.final_format_report.json
├── train.json
├── validation.json
└── test.json
```

#### Step 4 Report Summary

Initial Step 4 input:

| Quantity                          |  Count |
| --------------------------------- | -----: |
| Records entering final formatting | 22,272 |

Cleaning-stage statistics:

| Statistic                              | Count |
| -------------------------------------- | ----: |
| Invalid-answer records dropped         |   347 |
| Event lists reordered                  | 7,470 |
| Answers reordered by event-field order |   314 |
| Complex-to-medium relabeling           | 1,646 |

After cleaning but before level balancing:

| Level   |  Count |
| ------- | -----: |
| Simple  |  6,933 |
| Medium  |  7,445 |
| Complex |  7,547 |
| Total   | 21,925 |

After level balancing:

| Level   |  Count |
| ------- | -----: |
| Simple  |  6,933 |
| Medium  |  6,933 |
| Complex |  6,933 |
| Total   | 20,799 |

Final split sizes:

| Split      |  Count |
| ---------- | -----: |
| Train      | 16,639 |
| Validation |  2,080 |
| Test       |  2,080 |

These are the final TimelineCronQ-R splits used in the experiments.

---

### Step 5. LLM-Based Surface-Form Rewriting

The final stage applies constrained natural-language polishing to the finalized train, validation, and test splits.

It is intended to:

- rewrite malformed or template-like questions into more natural phrasing;
- preserve the original task semantics;
- preserve temporal operators and answer slots;
- repair answer temporal formatting only when an explicit correction is required.

This stage changes only surface wording.
It does not change the temporal program, split assignment, or benchmark composition established in Steps 1–4.
