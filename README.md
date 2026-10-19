# Name That LLM — my way up the leaderboard

Documenting my work on Kaggle's [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)
competition: given a short passage (40–250 words), predict which of 18 frontier LLMs wrote it.

## The task in short

- **Data:** `train.csv` has 2,699 passages for 150 prompts, and every prompt is answered by all 18 models. The hidden test set has ~4,000 passages from *unseen* prompts.
- **Metric:** balanced accuracy over 18 classes (chance ≈ 0.056).
- **Submission:** notebook-only. Kaggle re-runs the notebook on the hidden test set, with no internet. Output is `submission.parquet` with columns `id` and `model`.
- **Limits:** 5 submissions per day, up to 2 final submissions. The deadline is 13 Jan 2027.
- **Rules worth remembering:** external data and pretrained models are allowed if free and reasonably accessible to everyone. Public code sharing goes on Kaggle only.

## Results log

All CV numbers use 5-fold `GroupKFold` grouped by `prompt_id`. Random folds leak topic and mislead.

| Approach | Grouped CV | Public LB |
|---|---|---|
| Bigram LM (fwd + bwd, content words only) | 13–16% | — |
| Bigram LM (stopwords kept) | 18–19% | — |
| Hand-made style features only (punctuation, length, ...) | 27.5% | — |
| TF-IDF word 1–2-grams, logistic regression | 27.3% | — |
| TF-IDF char 2–5-grams, logistic regression | 32.7% | — |
| TF-IDF word+char+style, linear SVM (C=1, style weight 0.3) | **36.8%** | 34.8% |

What I learned:

- Removing stopwords hurt. Models differ more in function words and punctuation than in content words.
- Stacking bigram-LM scores on top of TF-IDF made things worse (35.0% and 33.3%, against 36.8% without).
- Errors are mostly within a model family: family-level accuracy is 69%. The Gemini flash models and the GPT-5.6 variants are the hardest to separate.
- Next idea: fine-tune a small transformer on a free Kaggle GPU.

## Repo layout

| Path | What it is |
|---|---|
| `data/` | competition files (not committed) |
| `bigram_lm.py` | forward/backward bigram language models per LLM |
| `tfidf_baseline.py` | TF-IDF and style-feature baselines with grouped CV |
| `tune_and_confuse.py` | tuning, LM stacking and the confusion analysis |
| `make_submission.py` | local fit and predict (a local file can't be submitted to this competition) |
| `kernel/` | the Kaggle notebook: `name_that_llm_tfidf.py` and its `kernel-metadata.json` |

## Submitting

```bash
cd kernel
kaggle kernels push -p .                       # upload + run the notebook
kaggle kernels status <user>/name-that-llm-tfidf-svm
kaggle competitions submit name-that-llm -k <user>/name-that-llm-tfidf-svm -f submission.parquet -v <version> -m "message"
```

## Pitfalls I hit

- **File uploads are rejected** (HTTP 400). The competition is notebook-only.
- **"Notebook Threw Exception" on the hidden rerun**, though the visible run passed. The hidden `test.csv` has ~4,000 rows instead of 3. My first script asserted that its ids matched `sample_submission.parquet`, which breaks if that file isn't regenerated. Now the ids come straight from `test.csv`, missing text is tolerated, and the script was tested against a fake 4,000-row test set before submitting. Don't make a notebook depend on anything that differs between the visible run and the rerun.
- A leftover `-I` flag stops Python importing sibling modules. Don't use it for scripts that import local files.
