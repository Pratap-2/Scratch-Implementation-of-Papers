# DeepML Notes

<p align="center">
  <strong>A hands-on notebook in code for machine learning, NLP, and Transformers.</strong>
  <br />
  Small implementations, experiments, and interview references — organized by topic.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-learning%20projects-3776AB?logo=python&logoColor=white" />
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-experiments-013243?logo=numpy&logoColor=white" />
  <img alt="Status" src="https://img.shields.io/badge/status-work%20in%20progress-orange" />
</p>

> **Learning repository, not a library:** most Python files are self-contained experiments. There is no shared package API, dependency lockfile, or promise that every script is production-ready.

## Start here

- **Learn the foundations:** [ML scratch](<ML scratch/>)
- **Explore a character-level GPT building block:** [Mini GPT](<Mini GPT/>)
- **Read the encoder–decoder implementation and run its tests:** [Transformer project](<Transformers(Attention_is_all_you_need)/README.md>)
- **Browse the interview PDFs:** [Interview Material](<Interview Material/>)

## Contents

1. [What's in the repository](#whats-in-the-repository)
2. [Learning paths](#learning-paths)
3. [Getting started](#getting-started)
4. [Running the Transformer tests](#running-the-transformer-tests)
5. [Notes and limitations](#notes-and-limitations)

## What's in the repository

```text
.
├── Interview Material/                  # ML interview question sets and handbook
├── ML scratch/                          # Classical ML, NLP, RAG, and attention exercises
│   ├── decision_trees/
│   ├── NLP/
│   └── Transformers/
├── Mini GPT/                            # Character tokens, embeddings, causal attention
└── Transformers(Attention_is_all_you_need)/
                                         # NumPy encoder–decoder and integration tests
```

### Classical machine learning

| Topic | Exercises |
| --- | --- |
| Regression | [Linear regression](<ML scratch/linear_regression.py>), [advanced linear regression](<ML scratch/adv_lr.py>) |
| Classification | [Logistic regression](<ML scratch/log_reg.py>), [multiclass classification](<ML scratch/multi_class_classification.py>) |
| Neural networks | [ANN / dense layers](<ML scratch/ann.py>) |
| Model evaluation | [K-fold cross-validation](<ML scratch/k-fold-cv.py>) |
| Decision trees | [Tree basics](<ML scratch/decision_trees/baiscs.py>), [entropy-based tree](<ML scratch/decision_trees/entropy_based_dt.py>), [Gini-based tree](<ML scratch/decision_trees/gini_gain_dt.py>), [regression tree](<ML scratch/decision_trees/regression_trees.py>) |
| Ensembles | [Random forests](<ML scratch/decision_trees/random_forests.py>) |

### NLP, retrieval, and attention

| Area | Exercises |
| --- | --- |
| NLP basics | [Tokenization](<ML scratch/NLP/basics.py>), [TF–IDF](<ML scratch/NLP/tf-idf.py>) |
| Retrieval-augmented generation | [Basic RAG pipeline](<ML scratch/rag_basic_se.py>), [PDF/LangChain RAG draft](<ML scratch/NLP/rag.py>), [retrieval metrics](<ML scratch/NLP/rag-metrics.py>) |
| Attention | [Self-attention](<ML scratch/Transformers/self-attention.py>), [multi-head attention](<ML scratch/Transformers/multi_head_attention.py>), [masked multi-head attention](<ML scratch/Transformers/masked_multi_head.py>), [optimized MHA experiment](<ML scratch/Transformers/optimized_MHA.py>), [positional encoding](<ML scratch/Transformers/positional_encoding.py>) |
| C++ experiment | [SHA source](<ML scratch/Transformers/sha.cpp>) |

### Mini GPT

The [Mini GPT folder](<Mini GPT/>) explores the input path and causal attention for a small character-level language model:

- [Character-level tokenization and batches](<Mini GPT/character_level_tokenization.py>)
- [Token and position embeddings](<Mini GPT/embeddings.py>)
- [Causal self-attention with forward/backward mechanics](<Mini GPT/casual attention.py>)
- [Additional attention experiment](<Mini GPT/try.py>)
- [Folder notes](<Mini GPT/readme.md>) and [reference PDF](<Mini GPT/PDFs/Resource_1.pdf>)

### Transformer encoder–decoder

[Transformers(Attention_is_all_you_need)](<Transformers(Attention_is_all_you_need)/README.md>) contains a NumPy implementation of encoder and decoder blocks, including positional encodings, masked decoder self-attention, cross-attention, residual connections, and layer normalization.

- [Encoder](<Transformers(Attention_is_all_you_need)/encoder.py>)
- [Decoder](<Transformers(Attention_is_all_you_need)/decoder.py>)
- [KV cache experiment](<Transformers(Attention_is_all_you_need)/KV-cache.py>)
- [Integration tests](<Transformers(Attention_is_all_you_need)/test_transformer.py>)
- [Implementation notes and limitations](<Transformers(Attention_is_all_you_need)/explaination.md>)

See the project README for its scope, usage example, and test details.

### Interview references

The [Interview Material folder](<Interview Material/>) contains:

- `Questions_ML_positions_1.pdf`
- `Questions_ML_positions_2.pdf`
- `ml_interview_handbook.pdf`

## Learning paths

**Classical ML** → regression and classification → decision trees → random forests → cross-validation

**Language models** → tokenization → embeddings → self-attention → causal attention → encoder–decoder Transformers

**Retrieval systems** → text splitting → vector similarity → top-*k* metrics → RAG pipeline

The paths are suggestions for browsing, not a required sequence. The exercises are separate files and do not share a common training framework.

## Getting started

Use Python 3.9 or newer and install NumPy for the NumPy-based examples:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Run a script from the repository root by passing its path in quotes (some directories and files contain spaces):

```bash
python "ML scratch/linear_regression.py"
python "Mini GPT/character_level_tokenization.py"
```

Some scripts only define functions or classes and will not print output when run directly. Check the source for its example or `__main__` block before expecting a demo.

## Running the Transformer tests

From the repository root:

```bash
python -m unittest discover -s "Transformers(Attention_is_all_you_need)" -p "test_*.py" -v
```

These tests cover selected forward-pass behavior and shape/masking checks for that folder. They do **not** test every script in the repository or establish that the Transformer is a complete, trainable text-to-text model. See the [Transformer project README](<Transformers(Attention_is_all_you_need)/README.md>) for details.

## Notes and limitations

- No root-level `requirements.txt`, `pyproject.toml`, or package installer is provided. Install dependencies for the individual experiment you want to run.
- NumPy is used by most examples. The PDF RAG draft additionally depends on LangChain integrations, FAISS, and a PDF loader, and expects an `OPENAI_API_KEY`; it is an experiment, not a configured turnkey application.
- Data is generally generated inline or represented by small examples. Bring your own datasets when adapting an exercise.
- File names preserve the current workspace, including spellings such as `baiscs.py`, `casual attention.py`, and `explaination.md`.
- The project-specific Transformer README documents further implementation boundaries, including the lack of padding masks, tokenization, training, and text generation in that encoder–decoder implementation.

---

<p align="center">
  Built as a growing collection of machine-learning notes and experiments.
</p>
