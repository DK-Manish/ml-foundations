# Student Score Predictor

A supervised machine learning program that predicts a student's average exam score based on demographic and background factors. It trains a Linear Regression model on real student data and evaluates how accurately it can predict scores for students it has never seen before.

---

## What the Program Does

1. Loads a dataset of 1,000 students with their exam scores and background details
2. Cleans and prepares the data for model training
3. Trains a Linear Regression model using 5 input features
4. Evaluates the model using MSE and RMSE on unseen test data
5. Displays a side-by-side comparison of predicted vs actual scores
6. Takes user input for a new student and predicts their expected average score
7. Saves all results to `results.txt`

---

## Dataset

- **Source**: [Kaggle — Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **File**: `StudentsPerformance.csv`
- **Records**: 1,000 students
- **Input features used**: gender, race/ethnicity, parental level of education, lunch type, test preparation course
- **Target (output)**: average score (mean of math, reading, and writing scores)

---

## Concepts Used

### Data Cleansing
Raw data is rarely perfect. Before training, the program:
- Checks for and removes any rows with missing values using `dropna()`
- Converts text/category columns (e.g. "male", "completed") into numbers using Label Encoding — ML models can only work with numbers, not words

### Train / Test Split
The dataset is split into two parts:
- **80% Training set** — the model learns from this data
- **20% Testing set** — used only to evaluate the model after training

This ensures the model is tested on data it has never seen, giving an honest measure of real-world performance. The analogy: you wouldn't test a student with the exact same questions they studied.

### Model Training (Linear Regression)
Linear Regression finds the best-fit line (or hyperplane) through the training data. It learns the relationship between input features and the target score by minimising the prediction error. The result is a formula of the form:

```
score = w1*gender + w2*race + w3*parental_edu + w4*lunch + w5*test_prep + bias
```

### Cost Function — Mean Squared Error (MSE)
MSE measures how far off the model's predictions are from the actual values:
- For each prediction: calculate `(predicted − actual)²`
- Average all those squared errors

Squaring ensures negative and positive errors don't cancel out, and penalises large errors more heavily. The model minimises MSE during training.

### RMSE (Root Mean Squared Error)
The square root of MSE. This brings the error back to the same unit as the scores (points out of 100), making it easy to interpret. An RMSE of 13.7 means predictions are off by roughly ±13.7 points on average.

---

## Libraries and Their Purpose

| Library | Purpose |
|---|---|
| `pandas` | Load and manipulate the CSV dataset as a structured table (DataFrame). Used for reading data, handling missing values, creating new columns, and filtering. |
| `scikit-learn` (`sklearn`) | The core machine learning library. Provides the `LinearRegression` model, `train_test_split` for splitting data, `mean_squared_error` for evaluation, and `LabelEncoder` for converting text categories to numbers. |
| `math` | Used for `math.sqrt()` to calculate RMSE from MSE. |
| `datetime` | Used to timestamp the `results.txt` output file with the date and time the program was run. |

---

## How to Run

```bash
python predictor.py
```

You will be prompted to enter values for a new student. The program will print the evaluation report and save results to `results.txt`.

---

## Output

- Console: dataset overview, cleansing info, train/test split, evaluation metrics, predicted vs actual table, and new student prediction
- File: `results.txt` — full run report saved automatically
