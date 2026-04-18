import pandas as pd
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

df = pd.read_csv(
    "spam.csv",
    encoding="latin-1",
    usecols=[0, 1],
    names=["label", "message"],
    header=0,
)

print("=" * 55)
print("        SPAM EMAIL CLASSIFIER")
print("=" * 55)
print(f"\n   Total messages : {len(df)}")
print(f"   Ham  (not spam): {(df['label'] == 'ham').sum()}")
print(f"   Spam           : {(df['label'] == 'spam').sum()}")


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_message"] = df["message"].apply(preprocess)
print("\n   Text preprocessing : done (lowercase + punctuation removed)")

df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})

X = df["clean_message"]
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    # stratify=y preserves the ham/spam ratio in both splits
)

print(f"\n   Training messages : {len(X_train)}")
print(f"   Testing messages  : {len(X_test)}")

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
# transform only — never fit on test data to avoid data leakage
X_test_vec = vectorizer.transform(X_test)

print("\n   Text vectorization  : TF-IDF (5 000 features)")

model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("   Model training      : Naive Bayes — done")

y_pred = model.predict(X_test_vec)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 55)
print("         PERFORMANCE METRICS")
print("=" * 55)
print(f"\n   Accuracy  : {accuracy * 100:.2f}%")
print(f"   Precision : {precision * 100:.2f}%")
print(f"   Recall    : {recall * 100:.2f}%")

print("\n   Confusion Matrix")
print("                  Predicted Ham   Predicted Spam")
print(f"   Actual Ham  :  {cm[0][0]:>13}   {cm[0][1]:>13}")
print(f"   Actual Spam :  {cm[1][0]:>13}   {cm[1][1]:>13}")

tn, fp, fn, tp = cm.ravel()
print(f"\n   True Negatives  (ham  correctly identified) : {tn}")
print(f"   False Positives (ham  wrongly flagged spam)  : {fp}")
print(f"   False Negatives (spam that slipped through)  : {fn}")
print(f"   True Positives  (spam correctly caught)      : {tp}")

test_messages = [
    "Congratulations! You have won a FREE ticket to Bahamas. Call now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account has been compromised. Click here to verify.",
    "Can you send me the notes from today's class?",
    "Win cash prizes! Text WIN to 80488 now. Free entry!",
]

print("\n" + "=" * 55)
print("      TEST WITH NEW UNSEEN MESSAGES")
print("=" * 55)

for msg in test_messages:
    cleaned    = preprocess(msg)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label      = "SPAM" if prediction == 1 else "HAM"
    print(f"\n   Message   : {msg[:60]}{'...' if len(msg) > 60 else ''}")
    print(f"   Prediction: {label}")

print()

results_lines = [
    "=" * 55,
    "        SPAM EMAIL CLASSIFIER — RESULTS",
    f"        Run date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 55,
    "",
    "DATASET SUMMARY",
    f"   Total messages : {len(df)}",
    f"   Ham  (not spam): {(df['label'] == 'ham').sum()}",
    f"   Spam           : {(df['label'] == 'spam').sum()}",
    "",
    "TRAIN / TEST SPLIT",
    f"   Training messages : {len(X_train)}",
    f"   Testing messages  : {len(X_test)}",
    "",
    "MODEL",
    "   Algorithm      : Multinomial Naive Bayes",
    "   Vectorization  : TF-IDF (5,000 features)",
    "",
    "=" * 55,
    "PERFORMANCE METRICS",
    "=" * 55,
    f"   Accuracy  : {accuracy * 100:.2f}%",
    f"   Precision : {precision * 100:.2f}%",
    f"   Recall    : {recall * 100:.2f}%",
    "",
    "CONFUSION MATRIX",
    "                  Predicted Ham   Predicted Spam",
    f"   Actual Ham  :  {cm[0][0]:>13}   {cm[0][1]:>13}",
    f"   Actual Spam :  {cm[1][0]:>13}   {cm[1][1]:>13}",
    "",
    f"   True Negatives  (ham  correctly identified) : {tn}",
    f"   False Positives (ham  wrongly flagged spam)  : {fp}",
    f"   False Negatives (spam that slipped through)  : {fn}",
    f"   True Positives  (spam correctly caught)      : {tp}",
    "",
    "=" * 55,
    "TEST WITH NEW UNSEEN MESSAGES",
    "=" * 55,
]

for msg in test_messages:
    cleaned    = preprocess(msg)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label      = "SPAM" if prediction == 1 else "HAM"
    results_lines.append(f"\n   Message   : {msg}")
    results_lines.append(f"   Prediction: {label}")

results_lines.append("")

with open("results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))

print("Results saved to results.txt")
