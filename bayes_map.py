import os
import sys
import argparse
import logging
from math import sqrt, exp, pi
import datetime
import time

def stripdt(src):
    dt = datetime.datetime.strptime(src, '%Y-%m-%d')
    return str(dt.month)

def training(instances, labels):
    summarize = summarize_by_class(instances, labels)
    return summarize

def predict(instance, parameters, priors):
    probabilities = calculate_class_probabilities(parameters, instance, priors)
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

    correct = 0
    for idx in range(len(predictions)):
        if predictions[idx] == answers[idx]:
            correct += 1
    accuracy = round(correct / len(answers), 2) * 100

    tp = 0
    fp = 0
    for idx in range(len(predictions)):
        if predictions[idx] == 1:
            if answers[idx] == 1:
                tp += 1
            else:
                fp += 1
    print(f"tp: {tp}, fp: {fp}")
    precision = round(tp / (tp + fp), 2) * 100

    tp = 0
    fn = 0
    for idx in range(len(answers)):
        if answers[idx] == 1:
            if predictions[idx] == 1:
                tp += 1
            else:
                fn += 1
    recall = round(tp / (tp + fn), 2) * 100
    
    f1 = 2 * (precision * recall) / (precision + recall)

    logging.info("accuracy: {}%".format(accuracy))
    logging.info("precision: {}%".format(precision))
    logging.info("recall: {}%".format(recall))
    logging.info("f1: {}".format(f1))
    
    return accuracy, precision, recall, f1

def load_csv(filename):
    instances = []
    labels = []
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(',')
            instances.append(tmp[:-1])
            labels.append(tmp[-1])
    for inst in instances:
        inst[0] = stripdt(inst[0])  
    logging.debug("instances: {}".format(instances))          
    return instances, labels

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(labels):
    unique = set(labels)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(labels)):
        labels[i] = lookup[labels[i]]
    return lookup

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)

def separate_by_class(dataset, labels):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = labels[i]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    return summaries

def summarize_by_class(dataset, labels):
    separated = separate_by_class(dataset, labels)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, row, priors):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = priors[class_value]
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

def preprocess(train_file, test_file):
    train_instances, train_labels = load_csv(train_file)
    for i in range(len(train_instances[0])):
        str_column_to_float(train_instances, i)
    str_column_to_int(train_labels)

    test_instances, test_labels = load_csv(test_file)
    for i in range(len(test_instances[0])):
        str_column_to_float(test_instances, i)
    str_column_to_int(test_labels)
    
    return train_instances, train_labels, test_instances, test_labels

def run(train_instances: list, train_labels: list, test_instances: list, test_labels: list):
    logging.debug("instances: {}".format(train_instances))
    logging.debug("labels: {}".format(train_labels))
    parameters = training(train_instances, train_labels)
    
    # Define priors (example: uniform priors)
    unique_labels = set(train_labels)
    priors = {label: 1/len(unique_labels) for label in unique_labels}
    
    predictions = []
    for instance in test_instances:
        result = predict(instance, parameters, priors)

        if result not in [0, 1]:
            logging.error("The result must be either 0 or 1")
            sys.exit(1)

        predictions.append(result)
    
    acc, prc, rec, f1 = report(predictions, test_labels)
    return acc, prc, rec, f1

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=False, metavar="<file path to the training dataset>", help="File path of the training dataset", default="./dataset/train/training.csv")
    parser.add_argument("-u", "--testing", required=False, metavar="<file path to the testing dataset>", help="File path of the testing dataset", default="./dataset/test/testing.csv")
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

    timer_start = time.time()
    data = preprocess(args.training, args.testing)
    run(*data)    
    timer_end = time.time()
    logging.info("Time elapsed: {}s".format(timer_end - timer_start))

if __name__ == "__main__":
    main()