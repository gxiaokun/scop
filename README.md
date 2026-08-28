# SCoP

> 🎉 Our paper has been accepted by CIKM 2026!

**SCoP: Structured Constraint Parsing for Evidence-Space Control in Temporal Knowledge Graph Question Answering**

This repository contains the code for the SCoP paper.
SCoP narrows the evidence space for temporal knowledge graph question answering before generating an answer, using both structural and temporal constraints.

![SCoP Architecture](Framework.png)

Supported datasets:

- `MultiTQ`
- `TimelineCronQR` in code, corresponding to **TimelineCronQ-R** in the paper

For details of the reconstructed TimelineCronQ-R benchmark, see [TimelineCronQ-R Reconstruction](./TimelineCronQ-R_repair_code/README.md).

## Environment Setup

We recommend the following environment:

- Python 3.11
- PyTorch 2.6.0+cu124
- CUDA 12.4

```bash
conda create -n scop python=3.11 -y
conda activate scop
pip install -r requirements.txt
```

## Repository Structure

```text
SCoP/
├── scop_main.py
├── timeline_baseline.py
├── run_scop.sh
├── run_timeline_baselines.sh
├── config.env
└── src/
    ├── scop.py
    ├── constraint.py
    ├── co_retriver.py
    ├── llm/
    ├── eval/
    ├── utils/
    └── config/
```

| Path                          | Description                                                                 |
| ----------------------------- | --------------------------------------------------------------------------- |
| `scop_main.py`              | Entry point for SCoP experiments                                            |
| `timeline_baseline.py`      | Entry point for TimelineCronQ-R baseline experiments                        |
| `run_scop.sh`               | Runs SCoP and its ablations                                                 |
| `run_timeline_baselines.sh` | Runs the TimelineCronQ-R baselines                                          |
| `src/scop.py`               | SCoP pipeline                                                               |
| `src/constraint.py`         | Parses and applies temporal constraints                                     |
| `src/co_retriver.py`        | Retrieves triples, coordinates alignment, and prepares evidence             |
| `src/llm/`                  | Prompts, few-shot examples, and structured-parsing modules                  |
| `src/eval/`                 | Hits@k evaluation and evidence-space analysis                               |
| `src/utils/`                | Graph construction, FAISS indexing, and shared utilities                    |
| `src/config/`               | Configuration loading and runtime settings                                  |

## Configuration

Runtime settings are read from `config.env`:

```bash
CHAT_MODEL=
OPENAI_BASE_URL=
OPENAI_API_KEY=

TEMPERATURE=0.0
TIMEOUT=30
MAX_TOKENS=8192

EMBED_MODEL_PATH=
EMBED_BATCH_SIZE=64
CUDA_DEVICE=cuda:0

DATASET_DIR=./data/datasets
BASE_STORE_DIR=./data/build_store
EXPERIMENT_OUTPUT_DIR=./test_run_results
```

Set the chat-model endpoint and API key, the local embedding-model path, and the dataset, artifact, and output directories before running an experiment.
The embedding model is loaded with `local_files_only=True`, so `EMBED_MODEL_PATH` must point to a locally available SentenceTransformer-compatible model.

---

## Dataset Layout

```text
DATASET_DIR/
├── MultiTQ/
│   ├── kg/full.txt
│   └── questions/test.json
└── TimelineCronQR/
    ├── kg/full.txt
    └── questions/test.json
```

`kg/full.txt` supports either point facts:

```text
subject<TAB>relation<TAB>object<TAB>yyyy-mm-dd
```

or interval facts:

```text
subject<TAB>relation<TAB>object<TAB>start_time<TAB>end_time
```

---

## Running SCoP

### MultiTQ

```bash
DATASET=MultiTQ \
TEST_SIZE=54584 \
ABLATION_TYPES="full" \
bash run_scop.sh
```

### TimelineCronQ-R / `TimelineCronQR`

```bash
DATASET=TimelineCronQR \
TEST_SIZE=2080 \
ABLATION_TYPES="full" \
bash run_scop.sh
```

On the first run, SCoP builds the temporal graph and FAISS retrieval artifacts under:

```text
BASE_STORE_DIR/<DATASET>/
```

---

## Ablation Studies

The following ablation modes are available:

```text
full
no_triple
no_align
no_constraint
```

### MultiTQ

```bash
DATASET=MultiTQ \
TEST_SIZE=54584 \
ABLATION_TYPES="full no_triple no_align no_constraint" \
bash run_scop.sh
```

### TimelineCronQR

```bash
DATASET=TimelineCronQR \
TEST_SIZE=2080 \
ABLATION_TYPES="full no_triple no_align no_constraint" \
bash run_scop.sh
```

---

## TimelineCronQR Baselines

The paper uses the following controlled baseline setting:

- final evidence budget: `20`
- pre-filter retrieval budget: `50`
- max iterative retrieval steps: `5`

```bash
TEST_SIZE=2080 \
FINAL_K=20 \
FILTER_RETRIEVE_K=50 \
MAX_STEPS=5 \
bash run_timeline_baselines.sh
```

Supported modes:

```text
rag
rag_filter
hyde_rag
hyde_rag_filter
query2doc_rag
query2doc_rag_filter
react
react_filter
ircot
ircot_filter
```

To run a subset of baselines:

```bash
MODE_LIST="rag rag_filter react react_filter" bash run_timeline_baselines.sh
```

---

## Outputs and Paper Results

SCoP outputs are written to:

```text
EXPERIMENT_OUTPUT_DIR/ablation/<DATASET>/<MODEL>/<ABLATION_TYPE>/test_<N>/run_<ID>/
```

The following files are particularly useful when inspecting a run:

```text
q_decomposed.json
aligned_decomposed.json
constrained.json
answer_file.json
```

Baseline outputs are written to:

```text
EXPERIMENT_OUTPUT_DIR/corn_baseline/
```

Evaluation runs automatically after SCoP finishes.
The reported metrics include Hits@1 and Hits@10, together with an evidence-space analysis based on the constrained outputs.
