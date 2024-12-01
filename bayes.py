import os
import sys
import argparse
import logging
from math import sqrt, exp, pi
import datetime

def stripdt(src):
    return datetime.datetime.splittime(src, '%y-%m-%d')

def training(instances, labels):
    summarize = summarize_by_class(instances, labels)
    return summarize

def predict(instance, parameters):
    probabilities = calculate_class_probabilities(parameters, instance)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def report(predictions, answers):
    if len(predictions) != len(answers):
        logging.error("The lengths of two arguments should be same")
        sys.exit(1)

    # accuracy
    correct = 0
    for idx in range(len(predictions)):
        if predictions[idx] == answers[idx]:
            correct += 1
    accuracy = round(correct / len(answers), 2) * 100

    # precision
    tp = 0
    fp = 0
    for idx in range(len(predictions)):
        if predictions[idx] == 1:
            if answers[idx] == 1:
                tp += 1
            else:
                fp += 1
    precision = round(tp / (tp + fp), 2) * 100

    # recall
    tp = 0
    fn = 0
    for idx in range(len(answers)):
        if answers[idx] == 1:
            if predictions[idx] == 1:
                tp += 1
            else:
                fn += 1
    recall = round(tp / (tp + fn), 2) * 100

    logging.info("accuracy: {}%".format(accuracy))
    logging.info("precision: {}%".format(precision))
    logging.info("recall: {}%".format(recall))

# Load a CSV file
def load_csv(filename):
    instances = []
    labels = []
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(',')
            instances.append(tmp[:-1])
            labels.append(tmp[-1])
    return instances, labels

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(labels):
    unique = set(labels)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(labels)):
        labels[i] = lookup[labels[i]]
    return lookup


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset, labels):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = labels[i]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    return summaries

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset, labels):
    separated = separate_by_class(dataset, labels)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


def run(train_file, test_file):
    # training phase
    instances, labels = load_csv(train_file)
    for i in range(len(instances[0])):
        str_column_to_float(instances, i)
    str_column_to_int(labels)

    logging.debug("instances: {}".format(instances))
    logging.debug("labels: {}".format(labels))
    parameters = training(instances, labels)

    # testing phase
    instances, labels = load_csv(test_file)
    for i in range(len(instances[0])):
        str_column_to_float(instances, i)
    str_column_to_int(labels)

    predictions = []
    for instance in instances:
        result = predict(instance, parameters)

        if result not in [0, 1]:
            logging.error("The result must be either 0 or 1")
            sys.exit(1)

        predictions.append(result)
    
    # report
    report(predictions, labels)

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=False, metavar="<file path to the training dataset>", help="File path of the training dataset", default="./dataset/train/training1.csv")
    parser.add_argument("-u", "--testing", required=False, metavar="<file path to the testing dataset>", help="File path of the testing dataset", default="./dataset/test/testing1.csv")
    parser.add_argument("-l", "--log", help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)", type=str, default="INFO")

    args = parser.parse_args()
    return args

def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error("The training dataset does not exist: {}".format(args.training))
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error("The testing dataset does not exist: {}".format(args.testing))
        sys.exit(1)

    run(args.training, args.testing)

if __name__ == "__main__":
    main()
