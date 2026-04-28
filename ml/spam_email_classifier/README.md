# Spam Email Classifier

A supervised machine learning program that automatically classifies SMS messages as either **spam** or **ham** (not spam). It trains a Naive Bayes classification model on real labelled messages and evaluates its performance using accuracy, precision, recall, and a confusion matrix.

---

## What the Program Does

1. Loads a dataset of 5,572 real SMS messages labelled as spam or ham
2. Cleans and preprocesses the raw text (lowercasing, removing punctuation)
3. Converts text messages into numerical vectors using TF-IDF
4. Trains a Multinomial Naive Bayes classification model
5. Evaluates the model using accuracy, precision, recall, and a confusion matrix
6. Tests the model against 5 brand new unseen messages
7. Saves all results to `results.txt`

---

## Dataset

- **Source**: [Kaggle — SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **File**: `spam.csv`
- **Records**: 5,572 messages (4,825 ham, 747 spam)
- **Columns used**: `v1` (label: ham/spam), `v2` (message text)

---

## Concepts Used

### Text Preprocessing
Raw text messages are noisy and inconsistent. Before training, each message is:
- **Lowercased** —> so "FREE" and "free" are treated as the same word
- **Stripped of punctuation and special characters** —> removes noise that adds no meaning
- **Whitespace normalised** —> multiple spaces collapsed into one

This step reduces vocabulary size and ensures the model focuses on meaningful content.

### Text Vectorization (TF-IDF)
ML models cannot read words, they need numbers. TF-IDF (Term Frequency–Inverse Document Frequency) converts each message into a row of numbers, one per word in the vocabulary.

- **TF (Term Frequency)**: how often a word appears in this specific message
- **IDF (Inverse Document Frequency)**: how rare that word is across all messages

A word that appears often in one message but rarely in others (e.g. "prize", "winner", "free") gets a high score. It is a strong signal. Common words like "the" or "is" appear everywhere and get a low score automatically. The top 5,000 most informative words are used as features.

### Model Training (Multinomial Naive Bayes)
Naive Bayes is the classic algorithm for text classification. It learns the probability of each word appearing in spam messages versus ham messages. At prediction time, it multiplies those probabilities together for every word in the message and picks the more likely class. It is fast, simple, and highly effective for text.

### Train / Test Split
- **80% Training set** —> the model learns word probabilities from this data
- **20% Testing set** —> held back to measure real-world performance

`stratify=y` ensures the spam/ham ratio is the same in both splits, preventing an imbalanced evaluation.

### Accuracy
The percentage of all messages (both ham and spam) that were classified correctly.

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

### Precision
Of all messages the model **labelled as spam**, what fraction were actually spam?

```
Precision = True Positives / (True Positives + False Positives)
```

High precision means few legitimate emails are wrongly flagged. It is important because losing a real email is costly.

### Recall
Of all messages that **were actually spam**, what fraction did the model catch?

```
Recall = True Positives / (True Positives + False Negatives)
```

High recall means fewer spam messages slip through to the inbox.

### Confusion Matrix
A 2×2 table summarising all four possible outcomes:

|  | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | True Negative (TN) | False Positive (FP) |
| **Actual Spam** | False Negative (FN) | True Positive (TP) |

- **TN** — ham correctly identified as ham
- **FP** — ham wrongly flagged as spam (false alarm)
- **FN** — spam that slipped through undetected
- **TP** — spam correctly caught

---

## Libraries and Their Purpose

| Library | Purpose |
|---|---|
| `pandas` | Load and manipulate the CSV dataset as a structured table (DataFrame). Used for reading data, selecting columns, and mapping labels to numbers. |
| `scikit-learn` (`sklearn`) | The core machine learning library. Provides `TfidfVectorizer` for text vectorization, `MultinomialNB` for the Naive Bayes model, `train_test_split` for splitting data, and `accuracy_score`, `precision_score`, `recall_score`, `confusion_matrix` for evaluation. |
| `re` | Python's built-in regular expressions library. Used in preprocessing to strip punctuation and normalise whitespace from raw message text. |
| `datetime` | Used to timestamp the `results.txt` output file with the date and time the program was run. |

---

## How to Run

```bash
python classifier.py
```

No user input required. The program runs fully automatically and prints results to the console.

---

## Output

- Console: dataset summary, preprocessing info, train/test split, performance metrics, confusion matrix, and predictions for 5 unseen messages
- File: `results.txt` — full run report saved automatically

---

## Results (Sample Run)

| Metric | Score |
|---|---|
| Accuracy | 96.05% |
| Precision | 100.00% |
| Recall | 70.47% |

**Precision of 100%** means zero legitimate emails were wrongly flagged as spam.
**Recall of 70.47%** means ~30% of spam slipped through the model errs on the side of caution to protect real messages.
