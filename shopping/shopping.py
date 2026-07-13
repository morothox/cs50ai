import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    evidence = []
    labels = []

    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)

        next(reader)

        for row in reader:
            row_evidence = []  # Hier sammeln wir die Werte für DIESEN Nutzer

            month_filter = {
                "Jan": 0,
                "Feb": 1,
                "Mar": 2,
                "Apr": 3,
                "May": 4,
                "June": 5,
                "Jul": 6,
                "Aug": 7,
                "Sep": 8,
                "Oct": 9,
                "Nov": 10,
                "Dec": 11,
            }

            for index, element in enumerate(row):
                if index == 17:
                    if element == "TRUE":
                        labels.append(1)
                    else:
                        labels.append(0)

                else:
                    # month filter
                    if index == 10:
                        row_evidence.append(month_filter[element])
                    # int filter
                    elif index in [0, 2, 4, 11, 12, 13, 14]:
                        row_evidence.append(int(element))
                    # visitor
                    elif index == 15:
                        if element == "Returning_Visitor":
                            row_evidence.append(1)
                        else:
                            row_evidence.append(0)
                    # weekend
                    elif index == 16:
                        if element == "FALSE":
                            row_evidence.append(0)
                        else:
                            row_evidence.append(1)

                    else:
                        row_evidence.append(float(element))

            evidence.append(row_evidence)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    # 2. Initialize and train the model (with k=3)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(evidence, labels)

    return knn


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    pos = 0
    neg = 0
    count_pos = 0
    count_neg = 0
    for i in range(len(predictions)):
        if labels[i] == 1:
            count_pos += 1
            if predictions[i] == labels[i]:
                pos += 1
        elif labels[i] == 0:
            count_neg += 1
            if predictions[i] == labels[i]:
                neg += 1

    sensitivity = pos / count_pos
    specificity = neg / count_neg
    res = (sensitivity, specificity)
    return res


if __name__ == "__main__":
    main()
