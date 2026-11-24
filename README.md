# Iris Flower Classification

A beginner machine learning project that predicts an Iris flower species from four measurements: sepal length, sepal width, petal length, and petal width.

## CIS134

Add your own internship ID here before submitting this repository:

`YOUR_INTERN_ID`

## How it works

- Loads the Iris dataset included with scikit-learn.
- Splits the data into training and test sets while keeping the species balanced.
- Scales the measurements and trains a logistic regression classifier.
- Prints test accuracy, a classification report, a confusion matrix, and an example prediction.

## Run it

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows, activate the environment with `.venv\\Scripts\\activate` instead.

## Dataset

The project uses scikit-learn's built-in Iris dataset, originally introduced by R. A. Fisher. It contains 150 observations across three Iris species.

## Tools

Python, scikit-learn, logistic regression, and a train/test split.
