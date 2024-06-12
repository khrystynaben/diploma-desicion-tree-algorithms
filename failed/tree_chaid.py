import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

threshold=0.05


def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0] 
    total_entr = 0
    
    for c in class_list: 
        total_class_count = train_data[train_data[label] == c].shape[0] 
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row) 
        total_entr += total_class_entr 
    
    return total_entr

def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
    
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]  
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count 
            entropy_class = - probability_class * np.log2(probability_class)  
        entropy += entropy_class
    return entropy

def calc_chi_square(train_data, feature_name, feature_value, label, class_list):
    contingency_table = pd.crosstab(train_data[feature_name], train_data[label])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2

def calc_info_gain(train_data, feature_name, label, class_list):
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in train_data[feature_name].unique():
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list)
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy 
        
    return calc_total_entropy(train_data, label, class_list) - feature_info 

def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) 
                                            
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  
        feature_info_gain = calc_info_gain(train_data, feature, label, class_list)
        if max_info_gain < feature_info_gain: 
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature

def generate_sub_tree(train_data, feature_name, label, class_list):
    tree = {} 
    for feature_value in train_data[feature_name].unique():
        print("!!!! ", feature_value)
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        chi_square = calc_chi_square(train_data, feature_name, feature_value, label, class_list)
        if chi_square > threshold:  
            tree[feature_value] = "?"
        else:
            assigned_to_node = False 
            for c in class_list: 
                class_count = feature_value_data[feature_value_data[label] == c].shape[0] 
                if class_count == feature_value_data.shape[0]:
                    tree[feature_value] = c 
                    assigned_to_node = True
            if not assigned_to_node: 
                tree[feature_value] = "?" 
    return tree

def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0: 
        max_info_feature = find_most_informative_feature(train_data, label, class_list) 
        print("max_info_feature",max_info_feature)
        tree = generate_sub_tree(train_data, max_info_feature, label, class_list)
        next_root = None
        print("prev_feature_value",prev_feature_value)
        if prev_feature_value is not None:  
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else: 
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        
        for node, branch in list(next_root.items()): 
            if branch == "?": 
                feature_value_data = train_data[train_data[max_info_feature] == node] 
                print("feature_value_data",feature_value_data)
                if feature_value_data.shape[0] != 0:  # Check if the filtered data is not empty
                    make_tree(next_root, node, feature_value_data, label, class_list) 
                else:  # Handle empty data here
                    next_root[node] = "None"  # or any other suitable constant






def chaid(train_data, label, threshold=0.05):  # Set your threshold value for significance
    tree = {} 
    class_list = train_data[label].unique() 
    make_tree(tree, None, train_data, label, class_list) 
    return tree

# Example usage:
# Example data
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Example usage of the build_tree function
label = 'PlayTennis'
class_list = ['Yes', 'No']
tree = chaid(data, label,  threshold)

# Print the built tree
print(tree)

