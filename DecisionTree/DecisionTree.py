import math
import numpy as np
from Node import Node
import pandas as pd


class DecisionTree():

    def __init__(self, training_filepath, label_pos_idx, split_opt = 'E', depth = 10, fix_tree = False, fix_label = None):

        # Load training and test files.
        self.df = self.load_file(training_filepath)
        self.label_pos_idx = label_pos_idx

        # Set the max depth.
        self.max_depth = depth

        # Fix the data with majority labels.
        if fix_label:
            self.UpdateEmptyDataMajority(fix_label)

        # Get the data to feed into the tree.
        num_features_idx, unique_feat, labels = self.count_unique_features()
        
        self.features_idx = unique_feat

        # Store features and examples to give to ID3 algorthm.
        self.attributes = list(range(0, len(num_features_idx)))
        self.examples = list(range(0, len(self.df.index)))
        

        # Store the method used to split the data.
        if split_opt == 'E':
            self.split_method = self.entropy_inf_gain
        elif split_opt == 'G':
            self.split_method = self.gini_index
        else:
            self.split_method = self.majority_err

        self.tree = self.ID3(self.examples, self.attributes, 0)
    
    def load_file(self, filepath):
        data = pd.read_csv(filepath)
        return data

    def ID3(self, S, attributes, depth, attr_name = None):

        # If attributes is empty or max depth reached, return a leaf node with most common label.

        if len(attributes) == 0 or depth == self.max_depth:
            label = self.most_common(S)
            new_node = Node(attribute_name = attr_name, label=label)
            return new_node

        # If all examples have the same label, return a leaf node with the label.
        elif self.label_all_same(S):
            label = self.df.iloc[S, self.label_pos_idx].unique()
            new_node = Node(attribute_name = attr_name, label=label)
            return new_node
            
        else:
            # Attribute in attributes that best splits S
            best_attr = self.best_split(S, attributes, self.split_method)
            
            # Create a root node for the tree.
            root = Node(attribute_name = attr_name, attribute_index = best_attr, label = "root node")

            # for each possible value v that A can take.
            attribute_df = self.df.iloc[S, best_attr]
            categories = attribute_df.unique()
    
            for category in categories:
            
                temp_s = (attribute_df == category)
                S_v = (temp_s[temp_s]).index.tolist() # Get subset of examples with A=v
                if len(S_v) > 0:
                    new_attributes = [n for n in attributes if n != best_attr] # Pass in list of attributes without 
                    new_node = self.ID3( S_v, new_attributes, depth + 1, attr_name = category)
                    root.add_node(new_node)
                    
                else: 
                    label = self.most_common( S)
                    new_node = Node(attribute_name = category, label = label)
                    root.add_node(new_node)
                
            return root


    # Returns an index of number of unique features for each column
    # And a list of lists with unique features.
    # And a list of labels.
    def count_unique_features(self):
        unique_feat_idx = []
        unique_feat = []
        for index in range(self.df.shape[1]):
            if index != self.label_pos_idx:
                unique_feat_idx.append(self.df.iloc[:, index].nunique() )
                temp_ls = self.df.iloc[:, index].unique()
                temp_ls.sort()
                unique_feat.append(temp_ls)
        labels = self.df.iloc[:, self.label_pos_idx].unique()
        
        return unique_feat_idx, unique_feat, labels

    # Check if all labels in the set are the same. 
    def label_all_same(self, S):
        num_diff_labels = self.df.iloc[S, self.label_pos_idx].nunique()
        return num_diff_labels <= 1

    # Get the label that is most common in the set.
    def most_common(self, S):
        labels = self.df.iloc[S, self.label_pos_idx].unique()
        val_counts = self.df.iloc[S, self.label_pos_idx].value_counts()
        
        max_label = labels[0]
        num_max_label = 0

        for label in labels:
            if val_counts[label] > num_max_label:
                max_label = label
        
        return max_label

    # Get the index of the best split. Info gain is functor passed to 
    # determine best split.
    def best_split(self, S, attributes, inf_gain):
        high_entropy = 0 # Initialize with invalid values.
        best_attr = -1
        for attribute in attributes:
            curr_entropy = inf_gain( S, attribute)
            if curr_entropy > high_entropy:
                high_entropy = curr_entropy
                best_attr = attribute
        return best_attr


    def log_entropy(self, top, bot):
        if top == 0 or bot == 0:
            return 0
        else:
            return math.log(top / bot, 2.0)
        
    def get_prediction(self, example):
        return self.tree.get_prediction(example)

    # Update all labels matching the null datatype with
    # majority category label of that column.
    def UpdateEmptyDataMajority(self, fix_label):
       
        # Iterate through each of the columns.
        for index in range(self.df.shape[1]):
            if index != self.label_pos_idx:
        
                # print("col: ", col)
                # print("type self df: ", type(selfcl.df))
                col = self.df.iloc[:, index]
                categories = col.unique()
                val_counts = col.value_counts()
                
                max_label = None
                num_max_cat = 0
             
                # Get the category with the majority. 
                for category in categories:
                    if category != fix_label and val_counts[category] > num_max_cat:
                        max_label = category 
               
                matching = col==fix_label

               
                self.df.iloc[:,index].mask(matching, max_label, inplace=True)           

    def entropy_inf_gain(self, S, attribute):
        # DFs containing column matching sent attribute and label col respectively.
        attribute_df = self.df.iloc[S, attribute]
        label_df = self.df.iloc[S, self.label_pos_idx]
    
        # List of categories in the attribute.
        categories = attribute_df.unique()
        # Possible labels for the examples.
        labels = label_df.unique()
        total_ex = float(len(S))
        
        #Calculate current entropy.
        curr_entropy = 0
    
        for label in labels:
            top = float((label_df == label).sum())
            curr_entropy -= (top / total_ex)*self.log_entropy(top, total_ex)
        
        # Calculate expected entropy.
        expected_entropy = 0
        for category in categories:
            total_in_cat = float((attribute_df==category).sum())
            
            cat_entropy = 0
            # Get entropy for each individual label for the category.
            for label in labels:

                top = ((attribute_df == category) & (label_df == label)).sum()

                cat_entropy -= (top / total_in_cat)*self.log_entropy(top, total_in_cat)
            
            expected_entropy += total_in_cat / total_ex * cat_entropy
        
        # IG = curr_entropy - expected_entropy.
        return curr_entropy - expected_entropy
        

    def gini_index(self, S, attribute):
        # DFs containing column matching sent attribute and label col respectively.
        label_df = self.df.iloc[S, self.label_pos_idx]
        
        # Possible labels for the examples.
        labels = label_df.unique()
        total_ex = float(len(S))
        
        #Calculate current entropy.
        total_p2 = 0
    
        for label in labels:
            p = float((label_df == label).sum()) / total_ex
            total_p2 -= p**2
        
        # IG = curr_entropy - expected_entropy.
        return 1 - total_p2


    def majority_err(self, S, attribute):
        # DFs containing column matching sent attribute and label col respectively.
        label_df = self.df.iloc[S, self.label_pos_idx]
    
        # Possible labels for the examples.
        labels = label_df.unique() 
        total_ex = float(len(S))
        
        #Store the min labels to calculate the majority err.
        min_labels = total_ex
    
        for label in labels:
            num_ex = float((label_df == label).sum())
            if num_ex < min_labels:
                min_labels = num_ex
        
        
        # IG = 1 - minimum labeled value / num_examples
        return 1 - min_labels / total_ex






