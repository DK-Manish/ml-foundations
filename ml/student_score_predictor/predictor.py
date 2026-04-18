import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import math
from datetime import datetime

df = pd.read_csv("StudentsPerformance.csv")

print("=" * 55)
print("        STUDENT SCORE PREDICTOR")
print("=" * 55)
print("\n📋 Dataset Overview")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")

print("\n🧹 Data Cleansing")

missing = df.isnull().sum().sum()
print(f"   Missing values found : {missing}")

df.dropna(inplace=True)
print(f"   Rows after cleaning  : {len(df)}")

df["average score"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3

encoder = LabelEncoder()
categorical_cols = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

print("   Categorical columns  : encoded to numbers")

features = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

X = df[features]
y = df["average score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Train / Test Split")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)
print(f"\n✅ Model trained (Linear Regression)")

y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print(f"\n📈 Evaluation Report")
print(f"   Mean Squared Error (MSE)  : {mse:.2f}")
print(f"   Root Mean Sq. Error (RMSE): {rmse:.2f}")
print(f"   → On average, predictions are off by ±{rmse:.1f} points")

print(f"\n🔍 Predicted vs Actual (first 10 test samples)")
print(f"   {'Actual':>10}  {'Predicted':>10}  {'Diff':>8}")
print(f"   {'-'*35}")
for actual, predicted in zip(list(y_test[:10]), y_pred[:10]):
    diff = predicted - actual
    print(f"   {actual:>10.1f}  {predicted:>10.1f}  {diff:>+8.1f}")

print(f"\n{'=' * 55}")
print("  PREDICT SCORE FOR A NEW STUDENT")
print(f"{'=' * 55}")

print("""
Answer the following (enter the number for your choice):

Gender
  0 = Female
  1 = Male
""")
gender = int(input("   Gender: "))

print("""
Race / Ethnicity
  0 = Group A
  1 = Group B
  2 = Group C
  3 = Group D
  4 = Group E
""")
race = int(input("   Race/Ethnicity: "))

print("""
Parental Level of Education
  0 = Associate's degree
  1 = Bachelor's degree
  2 = High school
  3 = Master's degree
  4 = Some college
  5 = Some high school
""")
parental_edu = int(input("   Parental Education: "))

print("""
Lunch
  0 = Free / Reduced
  1 = Standard
""")
lunch = int(input("   Lunch: "))

print("""
Test Preparation Course
  0 = Completed
  1 = None
""")
test_prep = int(input("   Test Prep: "))

new_student    = [[gender, race, parental_edu, lunch, test_prep]]
predicted_score = model.predict(new_student)[0]

print(f"\n{'=' * 55}")
print(f"  Predicted Average Score : {predicted_score:.1f} / 100")
print(f"{'=' * 55}\n")

gender_map   = {0: "Female", 1: "Male"}
race_map     = {0: "Group A", 1: "Group B", 2: "Group C", 3: "Group D", 4: "Group E"}
edu_map      = {0: "Associate's degree", 1: "Bachelor's degree", 2: "High school",
                3: "Master's degree",    4: "Some college",      5: "Some high school"}
lunch_map    = {0: "Free / Reduced", 1: "Standard"}
testprep_map = {0: "Completed", 1: "None"}

results_lines = [
    "=" * 55,
    "     STUDENT SCORE PREDICTOR — RESULTS",
    f"     Run date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 55,
    "",
    "DATASET SUMMARY",
    f"   Total rows : {df.shape[0]}",
    f"   Columns    : {df.shape[1]}",
    f"   Missing values found : {missing}",
    "",
    "TRAIN / TEST SPLIT",
    f"   Training samples : {len(X_train)}",
    f"   Testing samples  : {len(X_test)}",
    "",
    "MODEL",
    "   Algorithm : Linear Regression",
    "   Features  : gender, race/ethnicity, parental education, lunch, test preparation",
    "   Target    : average score (mean of math, reading, writing)",
    "",
    "=" * 55,
    "EVALUATION REPORT",
    "=" * 55,
    f"   Mean Squared Error (MSE)   : {mse:.2f}",
    f"   Root Mean Sq. Error (RMSE) : {rmse:.2f}",
    f"   On average, predictions are off by ±{rmse:.1f} points",
    "",
    "PREDICTED vs ACTUAL (first 10 test samples)",
    f"   {'Actual':>10}  {'Predicted':>10}  {'Diff':>8}",
    f"   {'-' * 35}",
]

for actual, predicted in zip(list(y_test[:10]), y_pred[:10]):
    diff = predicted - actual
    results_lines.append(f"   {actual:>10.1f}  {predicted:>10.1f}  {diff:>+8.1f}")

results_lines += [
    "",
    "=" * 55,
    "SAMPLE NEW STUDENT PREDICTION",
    "=" * 55,
    f"   Gender             : {gender_map.get(gender, gender)}",
    f"   Race / Ethnicity   : {race_map.get(race, race)}",
    f"   Parental Education : {edu_map.get(parental_edu, parental_edu)}",
    f"   Lunch              : {lunch_map.get(lunch, lunch)}",
    f"   Test Preparation   : {testprep_map.get(test_prep, test_prep)}",
    f"   Predicted Score    : {predicted_score:.1f} / 100",
    "",
]

with open("results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))

print("Results saved to results.txt")
