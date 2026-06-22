# Multilingual Health Question Answering in Low-Resource African Languages

Final project for Machine Learning Techniques I — Zindi Competition Entry

**Best public leaderboard score: 0.526504**  
**Final submissions selected:** Experiment 8 (0.517886) and Experiment 10 (0.526504)

---

## Project Overview

This project tackles the [Zindi Multilingual Health QA Challenge](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge), which requires building a system that answers health questions in low-resource African languages including Akan, Luganda, Swahili, and Amharic.

Two modeling techniques were explored across 10 experiments:

- **Neural generation** — fine-tuning mT5-small and mT5-base to generate answers directly
- **TF-IDF retrieval** — finding the most similar training question and returning its answer

Retrieval-based approaches consistently outperformed generation, with the best result combining a tuned character n-gram range (2,4) with a train+validation retrieval index.

---

## Repository Structure

```
multilingual-health-qa/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── multilingual_health_qa.ipynb      # Main experiment notebook (runnable on Colab)
│
├── submissions/
│   ├── submission_e1_dummy.csv
│   ├── submission_e2_zeroshot.csv
│   ├── submission_e3_mt5small.csv
│   ├── submission_e4_tfidf_global.csv
│   ├── submission_e5_tfidf_lang.csv
│   ├── submission_e6_top3_longest.csv
│   ├── submission_e7_top3_similar.csv
│   ├── submission_e8_trainval_combined.csv
│   ├── submission_e9_mt5base.csv
    └── submission_e10_ngram24_combined.csv

```

---

## Dataset

Data is from the Zindi competition and cannot be redistributed here.

To get the data:
1. Register at [zindi.africa](https://zindi.africa) and join the competition
2. Download the zip file from the Data tab
3. Place it in your Google Drive — the notebook will handle extraction

| File | Rows | Description |
|---|---|---|
| Train.csv | 29,815 | Questions + reference answers |
| Val.csv | 6,686 | Validation questions + answers |
| Test.csv | 2,618 | Test questions (no answers) |

**Languages covered:** Akan (Aka_Gha), Amharic (Amh_Eth), Luganda (Lug_Uga), Swahili (Swa_Ken), English from Uganda/Ghana/Ethiopia/Kenya

---

## Experiment Results

| Exp | Approach | Local ROUGE-1 | Zindi Public Score |
|---|---|---|---|
| 1 | Dummy Baseline | — | — |
| 2 | mT5-small Zero-Shot | — | — |
| 3 | mT5-small Fine-Tuned | — | 0.168657 |
| 4 | Global TF-IDF Retrieval | 0.4279 | 0.497641 |
| 5 | Language-Specific TF-IDF | 0.4209 | 0.447403 |
| 6 | Top-3 Retrieval, Longest Answer | 0.3713 | 0.491549 |
| 7 | Top-3 Retrieval, Most Similar | 0.4279 | 0.497641 |
| 8 | Train+Val Combined Retrieval | 0.4278 | 0.517886 |
| 9 | mT5-base Fine-Tuned | 0.0132* | 0.252231 |
| 10 | N-gram (2,4) + Train+Val Combined | 0.4361 | **0.526504** |

*Local ROUGE for Experiment 9 reflects a defective inference run; the corrected run produced the 0.252231 Zindi score — see report Discussion for details.

---

## How to Run

### Prerequisites

All experiments run on **Google Colab** with a T4 GPU (free tier).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/notebooks/multilingual_health_qa.ipynb)

### Steps

1. Open the notebook in Colab using the badge above
2. Connect to a T4 GPU runtime: **Runtime > Change runtime type > T4 GPU**
3. Mount your Google Drive and place the competition zip file at:
   ```
   /content/drive/MyDrive/multilingual-health-question-answering-in-low-resource-african-languages-challenge20260521-4309-nrr2a2.zip
   ```
4. Run cells in order — each experiment is clearly labelled with a markdown header

### Notes on reproducibility

- Random seed is fixed at `42` throughout (`random`, `numpy`, `torch`, and `Trainer`)
- TF-IDF experiments (Experiments 4-8, 10) run entirely on CPU and complete in under 5 minutes
- mT5-small fine-tuning (Experiment 3) takes approximately 70-90 minutes on T4
- mT5-base fine-tuning (Experiment 9) takes approximately 3 hours on T4
- Trained model weights are saved to Google Drive, not included in this repo due to size (mT5-base checkpoint is ~2.3GB). Contact the author for access if needed.

---

## Installation

```bash
pip install transformers peft rouge-score datasets evaluate sentencepiece accelerate scikit-learn pandas numpy
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

---

## Evaluation Metrics

The competition uses a weighted mean of three metrics:

| Metric | Weight | Description |
|---|---|---|
| ROUGE-1 F1 | 37% | Unigram word overlap |
| ROUGE-L F1 | 37% | Longest common subsequence |
| LLM-as-a-Judge | 26% | Factual accuracy and completeness scored 1-5 by an LLM |

Local evaluation uses ROUGE-1 and ROUGE-L only, with a whitespace tokenizer safe for non-Latin scripts (Amharic's Ge'ez script).

---

## Key Findings

- Simple TF-IDF retrieval outperformed fine-tuned neural generation by nearly 3x (0.527 vs 0.169), suggesting the test set has strong lexical overlap with the training corpus
- Global retrieval (single index across all languages) consistently beat language-specific retrieval, as larger pools improve nearest-neighbor quality even across language boundaries
- Increasing the retrieval pool from train-only to train+val improved the score from 0.498 to 0.518
- Tuning the TF-IDF character n-gram range from (3,5) to (2,4) further improved to 0.527, likely benefiting morphologically rich languages like Akan and Luganda
- mT5-base outperformed mT5-small (0.252 vs 0.169) but neither approached the retrieval baseline

---

## Author

Emmanuella Briggs 
Machine Learning Techniques I — Final Project  
June 2026
