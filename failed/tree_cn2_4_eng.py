import pandas as pd
from collections import defaultdict
import math

# Provided dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Classes and labels
label = 'PlayTennis'
class_list = ['Yes', 'No']

def entropy(data, label):
    label_counts = data[label].value_counts()
    total_samples = len(data)
    entropy_val = 0
    for cls in class_list:
        if cls in label_counts:
            prob_cls = label_counts[cls] / total_samples
            entropy_val -= prob_cls * math.log(prob_cls, 2)
    return entropy_val

def information_gain(data, attr, label):
    attr_values = data[attr].unique()
    total_samples = len(data)
    entropy_before = entropy(data, label)
    gain = 0
    for value in attr_values:
        subset = data[data[attr] == value]
        prob_value = len(subset) / total_samples
        entropy_value = entropy(subset, label)
        gain += prob_value * entropy_value
    return entropy_before - gain

def find_best_split(data, label, class_list, min_info_gain=0.05, max_depth=5, depth=0):
    if depth >= max_depth:
        return None

    best_attr = None
    best_value = None
    best_info_gain = min_info_gain
    for attr in data.columns:
        if attr != label:
            values = data[attr].unique()
            for value in values:
                subset_with_attr_value = data[data[attr] == value]
                subset_without_attr_value = data[data[attr] != value]
                info_gain = information_gain(data, attr, label)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_attr = attr
                    best_value = value
                    best_subset = subset_with_attr_value

    if best_attr is None:
        return None

    tree = defaultdict(dict)
    tree[best_attr][best_value] = find_best_split(best_subset, label, class_list, min_info_gain, max_depth, depth+1)
    return tree

decision_tree = find_best_split(data, label, class_list, min_info_gain=0.02, max_depth=10)
print(decision_tree)
