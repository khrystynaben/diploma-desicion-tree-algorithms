import pandas as pd
from collections import defaultdict
import math

# Нова вибірка даних
data = pd.DataFrame({
    'Age': [25, 35, 45, 28, 42, 50, 32, 48, 22, 38],
    'Income': ['$50K', '$70K', '$60K', '$45K', '$80K', '$90K', '$55K', '$85K', '$40K', '$75K'],
    'Education': ['High School', 'Bachelor', 'Master', 'High School', 'PhD', 'Bachelor', 'Master', 'PhD', 'High School', 'Bachelor'],
    'Employment': ['Full-Time', 'Part-Time', 'Full-Time', 'Unemployed', 'Full-Time', 'Part-Time', 'Full-Time', 'Full-Time', 'Part-Time', 'Full-Time'],
    'LoanApproved': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes']
})

# Задані класи та мітки
label = 'LoanApproved'
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

def cn2(data, label, class_list):
    tree = defaultdict(dict)
    best_entropy = 0

    while len(data) > 0:
        current_entropy = entropy(data, label)
        if best_entropy == 0 or current_entropy < best_entropy:
            best_entropy = current_entropy
            best_attr = None
            best_value = None
            best_subset = None

        for attr in data.columns:
            if attr != label:
                values = data[attr].unique()
                for value in values:
                    subset_with_attr_value = data[data[attr] == value]
                    subset_without_attr_value = data[data[attr] != value]
                    info_gain = current_entropy - (len(subset_with_attr_value) / len(data) * entropy(subset_with_attr_value, label)
                                                    + len(subset_without_attr_value) / len(data) * entropy(subset_without_attr_value, label))
                    if info_gain > best_entropy:
                        best_entropy = info_gain
                        best_attr = attr
                        best_value = value
                        best_subset = subset_with_attr_value

        if best_attr is not None:
            tree[best_attr][best_value] = {}
            data = best_subset
        else:
            break

    return tree

decision_tree = cn2(data, label, class_list)
print(decision_tree)
