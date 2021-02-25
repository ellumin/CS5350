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

    print("accuracy: ", num_correct / test_ex_tot)

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


filename = "submission_test"
training_filepath_1 = "C:/Users/keato/OneDrive/Documents/CS5350/DecisionTree/bank/train.csv"
test_filepath_1 = "C:/Users/keato/OneDrive/Documents/CS5350/DecisionTree/bank/test.csv"
label_pos_idx_1 = 15
id_label = 'id'
pred_label = 'quality'
tree = Tree(training_filepath_1, label_pos_idx_1, 'E', 3)
create_prediction_file(tree, test_filepath_1, label_pos_idx_1, filename, id_label, pred_label)
get_accuracy_from_tree(tree, test_filepath_1, label_pos_idx_1)

# Car simulation.
# filename = "submission_test"
# training_filepath_1 = "C:/Users/keato/OneDrive/Documents/CS5350/DecisionTree/car/train.csv"
# test_filepath_1 = "C:/Users/keato/OneDrive/Documents/CS5350/DecisionTree/car/test.csv"
# label_pos_idx_1 = 6
# id_label = 'id'
# pred_label = 'quality'
# tree = Tree(training_filepath_1, label_pos_idx_1, 'E')
# create_prediction_file(tree, test_filepath_1, label_pos_idx_1, filename, id_label, pred_label)
# get_accuracy_from_tree(tree, test_filepath_1, label_pos_idx_1)

   
# filename = "submission_1"
# training_filepath = "G:/Main/School/College/UniversityOfUtah/FinalPush/MachineLearning/income-level-prediction-2021s/train_final.csv"
# test_filepath = "G:/Main/School/College/UniversityOfUtah/FinalPush/MachineLearning/income-level-prediction-2021s/test_final.csv"
# label_pos_idx = 14
# id_label = 'id'
# pred_label = 'Prediction'

# tree = Tree(training_filepath, label_pos_idx)
# create_prediction_file(tree, test_filepath, label_pos_idx, filename, id_label, pred_label)