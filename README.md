# Political Interview Response Clarity Classification

University project for the Artificial Intelligence course.

Given a question from a presidential press conference and the answer that was given, the model classifies the answer as **Clear Reply**, **Ambivalent** or **Clear Non-Reply**.

## Approach

1. Exploratory data analysis (class balance, text lengths)
2. Text cleaning, then question and answer joined into one text
3. Train/validation split grouped by answer (the same answer appears in several rows)
4. Two feature types: **TF-IDF** (unigrams + bigrams) and **Word2Vec** (averaged word vectors)
5. **Logistic Regression** tuned with grid search (`C`, `class_weight`) on macro F1
6. Comparison with a majority-class baseline, confusion matrix, final evaluation on the test set

## Results

| Model | Validation macro F1 |
|---|---|
| Majority class (always "Ambivalent") | 0.26 |
| Word2Vec + Logistic Regression (tuned) | 0.43 |
| **TF-IDF + Logistic Regression (tuned)** | **0.45** |

Final TF-IDF model on the test set: **macro F1 0.44**, accuracy 0.51.

The classes are imbalanced (about 60% "Ambivalent"), so models are compared with macro F1 rather than accuracy.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook clarity_classification.ipynb
```

The dataset ([QEvasion](https://huggingface.co/datasets/ailsntua/QEvasion)) is downloaded automatically from Hugging Face.

## Tech

Python, pandas, scikit-learn, gensim, matplotlib, seaborn
