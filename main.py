"""Beginner-friendly Iris flower classifier using scikit-learn."""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # Example: sepal length/width and petal length/width, in centimeters.
    example = [[5.1, 3.5, 1.4, 0.2]]
    predicted_species = iris.target_names[model.predict(example)[0]]
    print(f"\nExample flower prediction: {predicted_species}")


if __name__ == "__main__":
    main()
