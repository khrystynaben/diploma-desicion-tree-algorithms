#not works 
'''
  File "d:\university\diploma\new_chaid.py", line 14, in get_observed_expected_freq
    observed_freq = Counter(data[attribute])
  File "c:\Users\HP admin\AppData\Local\Programs\Python\Python39\lib\site-packages\pandas\core\frame.py", line 3807, in __getitem__
    indexer = self.columns.get_loc(key)
  File "c:\Users\HP admin\AppData\Local\Programs\Python\Python39\lib\site-packages\pandas\core\indexes\base.py", line 3804, in get_loc
    raise KeyError(key) from err
KeyError: 'math'
'''

from collections import Counter
import pandas as pd
import math

def calculate_chi_square(observed, expected):
    chi_square = 0.0
    for obs, exp in zip(observed, expected):
        if exp != 0:
            chi_square += ((obs - exp) ** 2) / exp
    return chi_square

def chaid(data, target_column):
    def get_observed_expected_freq(data, attribute, target_values):
        observed_freq = Counter(data[attribute])
        expected_freq = {}
        total_observed = len(data[attribute])
        
        for value in target_values:
            expected_freq[value] = (len(data[data[target_column] == value]) / total_observed) * observed_freq[value]
        
        return observed_freq, expected_freq
    
    def calculate_attribute_chi_square(observed_freq, expected_freq):
        chi_square_values = {}
        for key in observed_freq.keys():
            if key in expected_freq:
                chi_square_values[key] = calculate_chi_square(
                    observed_freq[key], expected_freq[key]
                )
        return chi_square_values
    
    def find_best_attribute(data, attributes, target_values):
        best_attribute = None
        best_chi_square = float('inf')
        
        for attribute in attributes:
            observed_freq, expected_freq = get_observed_expected_freq(data, attribute, target_values)
            chi_square_values = calculate_attribute_chi_square(observed_freq, expected_freq)
            attribute_chi_square = sum(chi_square_values.values())
            
            if attribute_chi_square < best_chi_square:
                best_chi_square = attribute_chi_square
                best_attribute = attribute
        
        return best_attribute
    
    def split_data(data, attribute):
        groups = data.groupby(attribute)
        return groups
    
    def build_tree(data, attributes, target_values, node=None):
        if node is None:
            node = {}
        
        if len(set(data[target_column])) == 1:
            return {'label': data[target_column].iloc[0]}
        
        best_attribute = find_best_attribute(data, attributes, target_values)
        node['attribute'] = best_attribute
        node['children'] = {}
        
        groups = split_data(data, best_attribute)
        for value, group in groups:
            node['children'][value] = build_tree(group.drop(columns=[best_attribute]), attributes, target_values)
        
        return node
    
    
    
    attributes = data.columns.tolist()
    attributes.remove(target_column)
    target_values = data[target_column].unique()
    
    decision_tree = build_tree(data, attributes, target_values)
    return decision_tree


def predict(row, tree):
        if 'label' in tree:
            return tree['label']
        else:
            attribute = tree['attribute']
            value = row[attribute]
            if value in tree['children']:
                child_tree = tree['children'][value]
                return predict(row, child_tree)
            else:
                return None
    
def accuracy(data, tree):
    correct = 0
    total = len(data)
    for index, row in data.iterrows():
        prediction = predict(row, tree)
        if prediction == row[target_column]:
            correct += 1
    return correct / total * 100
# Example data
data0 = pd.DataFrame({
    'math': ['A', 'C', 'D', 'B'],
    'physics': ['A', 'C', 'A', 'C'],
    'chemistry': ['A', 'A', 'D', 'A'],
    'additional': [5, 3, 1, 6],
    'attendance': ['at_yes', 'at_yes', 'at_yes', 'at_yes'],
    'mark': ['yes', 'no', 'no', 'yes']
})
test_data0 = pd.DataFrame({
    'math': ['A', 'C', 'D', 'B'],
    'physics': ['A', 'C', 'A', 'C'],
    'chemistry': ['A', 'A', 'D', 'A'],
    'additional': [5, 3, 1, 6],
    'attendance': ['at_yes', 'at_yes', 'at_yes', 'at_yes'],
    'mark': ['yes', 'no', 'no', 'yes']
})
# Target column
target_column = 'mark'

train_data_marks = pd.read_csv("D:/train/new_tree_data_marks.csv") 
test_data_marks = pd.read_csv("D:/test/new_tree_data_marks.csv")

print("Train Data Columns:")
print(train_data_marks.columns)
print("Test Data Columns:")
print(test_data_marks.columns)

#print(data0)
#print(train_data_marks)
# Run CHAID algorithm and calculate accuracy
decision_tree = chaid(train_data_marks, target_column)
print("Decision Tree:")
print(decision_tree)
print("Accuracy:", accuracy(test_data_marks, decision_tree))
