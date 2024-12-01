from bayes import run, str_column_to_float, str_column_to_int, stripdt
import itertools
import time
import os 

train_file = 'dataset/train/training.csv'
test_file = 'dataset/test/testing.csv'

def generate_subsets(src:list) -> list[list]:
    subsets = []
    for i in range(1, len(src)):
        for subset in itertools.combinations(src, i):
            subsets.append(list(subset))
    return [subset for subset in subsets if len(subset) > 0]

train_col_idx , test_col_idx = None, None

with open(train_file) as f:
    train_col_idx = f.readline().strip().split(',')[:-1]
with open(test_file) as f:
    test_col_idx = f.readline().strip().split(',')[:-1]

train_sets = generate_subsets(train_col_idx)
test_sets = generate_subsets(test_col_idx)

def load_csv(filename, pick_cols: list):
    instances = []
    labels = []
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(',')
            features = tmp[:-1]
            instance = [features[i] for i in pick_cols]
            instances.append(instance)
            labels.append(tmp[-1])
    # Apply stripdt to the first column of instances if pick_cols[0] == 0
    if pick_cols[0] == 0:
        for inst in instances:
            inst[0] = stripdt(inst[0])
            
    print("instances: {}".format(instances))
    return instances, labels

def pick(file, target:list): 
    target_cols_idx = []
    with open(file) as f:
        title = f.readline().strip().split(',')
        for i in range(len(title)):
            if title[i] in target:
                target_cols_idx.append(i)
    return target_cols_idx
    
results = []

try:
    # train_sets and test_sets are synchronized
    for i in range(len(train_sets)):
        train_sets_processed = pick(train_file, train_sets[i])
        test_sets_processed = pick(test_file, test_sets[i])
        train_instances, train_labels = load_csv(train_file, train_sets_processed)
        for i in range(len(train_instances[0])):
            str_column_to_float(train_instances, i)
        str_column_to_int(train_labels)
        test_instances, test_labels = load_csv(test_file, test_sets_processed)
        for i in range(len(test_instances[0])):
            str_column_to_float(test_instances, i)
        str_column_to_int(test_labels)
            
        acc, prc, rec, f1 = run(train_instances, train_labels, test_instances, test_labels)
        results.append((i, train_sets[i], test_sets[i], acc, prc, rec, f1))
except Exception as e:
    os.system('python automate.py & ')
    exit()
    
results.sort(key=lambda x: x[3], reverse=True)
print(f"\n\nTop 3 results: {results[:3]}")