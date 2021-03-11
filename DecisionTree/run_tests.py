import pandas as pd
import numpy as np

from DecisionTree import DecisionTree as Tree


def load_file(filepath):
    data = pd.read_csv(filepath)
    return data

def get_accuracy_from_tree(tree, test_filepath, label_pos_idx):
    
    # Get list of all examples to test on.
    test_ls = load_file(test_filepath).values.tolist()
    test_ex_tot = len(test_ls)
   
    num_correct = 0

    for example in test_ls:
        pred = tree.get_prediction(example)
    
        if example[label_pos_idx] == pred:
            num_correct += 1

    return num_correct / test_ex_tot

def create_prediction_file(tree, test_filepath, label_pos_idx, filename, id_label, pred_label):
    
    # Get list of all examples to test on.
    test_ls = load_file(test_filepath).values.tolist()
    test_ex_tot = len(test_ls)

    id_list = range(1, test_ex_tot+1)
        
    predictions = []
    for example in test_ls:
        pred = tree.get_prediction(example)
        if type(pred) is np.ndarray:
            pred = str(pred.tolist()[0])
        predictions.append(pred)
    
    submission = pd.DataFrame({id_label: id_list, pred_label: predictions})

    submission.to_csv(filename + '.csv', index=False)

def run_full_tree_test(training_filepath, test_filepath, label_pos_idx, max_depth, fix_tree = False, fix_label = None):
    info_split_types = ['E', 'G', 'M']
    
    for split_type in info_split_types:
        average_train_err = 0
        average_test_err = 0
        count = 0
        for depth in range(1, max_depth + 1):
            tree = Tree(training_filepath, label_pos_idx, split_type, depth, fix_tree, fix_label) 
            train_err = 1 - get_accuracy_from_tree(tree, training_filepath, label_pos_idx)
            test_err = 1 - get_accuracy_from_tree(tree, test_filepath, label_pos_idx)
            print("Type of split: ", split_type, "Depth: ", depth, "Train error: ", train_err, "Test error: ", test_err)
            average_train_err += train_err
            average_test_err += test_err
            count += 1
        average_train_err /= count
        average_test_err /= count
        print("Type of split: ", split_type, "Average_train err: ", average_train_err, "Average test err: ", average_test_err)

# # Car simulation.
filename = "submission_test"
training_filepath = "C:/Users/keato/OneDrive/Documents/CS5350/DecisionTree/car/train.csv"
test_filepath = "C:/Users/keato/OneDrive/Documents/CS5350/DecisionTree/car/test.csv"
label_pos_idx = 6
print("Car data: ")
run_full_tree_test(training_filepath, test_filepath, label_pos_idx, 6)


filename = "submission_test"
training_filepath = "C:/Users/keato/OneDrive/Documents/CS5350/DecisionTree/bank/train.csv"
test_filepath = "C:/Users/keato/OneDrive/Documents/CS5350/DecisionTree/bank/test.csv"
label_pos_idx = 16
print("Bank data using unknown as an attribute: ")
run_full_tree_test(training_filepath, test_filepath, label_pos_idx, 16)

filename = "submission_test"
training_filepath = "C:/Users/keato/OneDrive/Documents/CS5350/DecisionTree/bank/train.csv"
test_filepath = "C:/Users/keato/OneDrive/Documents/CS5350/DecisionTree/bank/test.csv"
label_pos_idx = 16
print("Bank data using majority instead of unknown")
run_full_tree_test(training_filepath, test_filepath, label_pos_idx, 16, True, "unknown")



