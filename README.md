# Intelligent Question Answering System for English Documents

An NLP system that answers English questions about a document by **retrieving** the most relevant passages and **extracting** the exact answer span from them, or saying that no reliable answer exists.

> Status: **M1 — data and splits.** This README grows with each milestone.

## Approach

1. **Retriever** — ranks document passages by relevance to the question (TF-IDF, BM25, GloVe, Sentence-BERT are compared).
2. **Reader** — a DistilBERT model fine-tuned on SQuAD 2.0 extracts the answer span.
3. **Confidence gate** — abstains when the answer score is below a threshold tuned on validation data.

No generative models or external LLM APIs are used.

## Repository structure

```
configs/       experiment settings (YAML)
data/          datasets and documents - not committed, see data/README.md
notebooks/     exploration and Colab runners
src/qa/        the Python package: all NLP logic lives here
experiments/   scripts that run experiments + small result files
app/           Gradio interface (calls src/qa only)
tests/         pytest unit tests
reports/       figures and methodology notes
```

## Setup

### Local (Windows)

```powershell
git clone https://github.com/AyaOuertani/intelligent-qa.git
cd intelligent-qa
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
```

### Google Colab

Open `notebooks/00_colab_setup.ipynb` and run all cells.

## Data

Primary dataset: **SQuAD 2.0** ([Rajpurkar et al., 2018](https://arxiv.org/abs/1806.03822)), downloaded with Hugging Face `datasets`.

| Split | Source | Use |
| --- | --- | --- |
| train | SQuAD 2.0 train, 90% of articles | Fine-tuning the reader |
| validation | SQuAD 2.0 train, 10% of articles (seed 42) | All tuning decisions |
| test | Official SQuAD 2.0 dev set | Final results only |

The split is by article and is fixed in `configs/squad_v2_article_splits.json`. Rebuild it with `python experiments/m1_make_splits.py`.

## Milestones

- [x] M0 — Repository and environment
- [x] M1 — Data loading, exploration and splits
- [ ] M2 — Preprocessing and segmentation
- [ ] M3 — Evaluation framework
- [ ] M4 — Experiment A: TF-IDF
- [ ] M5 — Experiment B: BM25
- [ ] M6 — Experiment C: semantic retrieval
- [ ] M7 — Experiment D: reader fine-tuning
- [ ] M8 — Experiment E: end-to-end pipeline
- [ ] M9 — Document ingestion
- [ ] M10 — Gradio interface
- [ ] M11 — Final evaluation and release
