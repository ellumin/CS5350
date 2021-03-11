import numpy as np
import random

class Node:
    def __init__(self, attribute_index = -1, attribute_name = None, label = "into node"):
        self.nodes = []
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.label = label

    def set_attribute_name(self, name):
        self.attribute_name = name

    def add_node(self, node):
        self.nodes.append(node)

    def get_prediction(self, example):
        if self.attribute_index == -1:
               return self.label
        
        for node in self.nodes:
            if example[self.attribute_index] == node.attribute_name:
                return node.get_prediction(example)
        
        # Get majority vote if base cases don't work.
        # votes = {}
        # votes = self.get_majority(votes)
        # votes = sorted(votes.items(), key=lambda x: x[1], reverse = True)
    
        # max_val = list(votes)[0][1]
        
        # labels = []
     
        # for vote in votes:
        #     if max_val == vote[1]:
        #         labels.append(vote[0])
     
        # # Return random selection of max voted labels.
        # return labels[random.randint(0, len(labels)-1)]

        # If there is a tie, pick one at random.
        
        for node in self.nodes:
            if node.attribute_index == -1:
                return node.label
    
    # returns the largest label vote.
    def get_majority(self, votes_dict):
        for node in self.nodes:
            if node.attribute_index == -1:
                key = node.label
                if type(key) is np.ndarray:
                    key = str(key.tolist()[0])
                if key in votes_dict:
                    votes_dict[key] += 1
                else:
                    votes_dict[key] = 1
                return votes_dict
            else:
                return node.get_majority(votes_dict)

    def __str__(self):
        nodes = [str(self.attribute_name)]
        if len(self.nodes) == 0 and self.label is None:
            nodes.append(str(self.label))
        else:
            for node in self.nodes:
                nodes.append("\t"+ str(node))
        
        return ''.join(nodes)
