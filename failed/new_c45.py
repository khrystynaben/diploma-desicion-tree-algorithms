import numpy as np
import pandas as pd
import time

def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0]
    total_entr = 0
    
    for c in class_list: 
        total_class_count = train_data[train_data[label] == c].shape[0]
        if total_class_count != 0:
            total_class_entr = - (total_class_count/total_row) * np.log2(total_class_count/total_row)
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

def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list)
        feature_value_probability = feature_value_count / total_row
        feature_info += feature_value_probability * feature_value_entropy
    
    return calc_total_entropy(train_data, label, class_list) - feature_info

def calc_info_gain_continuous(feature_name, train_data, label, class_list):
    total_row = train_data.shape[0]
    sorted_data = train_data.sort_values(by=feature_name)
    unique_values = sorted_data[feature_name].unique()
    
    if len(unique_values) == 1:
        return 0, None
    
    max_info_gain = -1
    best_split = None

    for i in range(1, len(unique_values)):
        split_value = (unique_values[i-1] + unique_values[i]) / 2
        left_split = sorted_data[sorted_data[feature_name] <= split_value]
        right_split = sorted_data[sorted_data[feature_name] > split_value]
        
        left_entropy = (left_split.shape[0] / total_row) * calc_entropy(left_split, label, class_list)
        right_entropy = (right_split.shape[0] / total_row) * calc_entropy(right_split, label, class_list)
        
        info_gain = calc_total_entropy(train_data, label, class_list) - (left_entropy + right_entropy)
        
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_split = split_value
            
    return max_info_gain, best_split

def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label)
    max_info_gain = -1
    max_info_feature = None
    best_split = None
    
    for feature in feature_list:  
        if train_data[feature].dtype == 'object':  # Категоріальні дані
            feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
            split_value = None
        else:  # Неперервні дані
            feature_info_gain, split_value = calc_info_gain_continuous(feature, train_data, label, class_list)
        
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature
            best_split = split_value
            
    return max_info_feature, best_split

def generate_sub_tree(feature_name, train_data, label, class_list, split_value=None):
    if split_value is None:
        feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
        tree = {} 
        
        for feature_value, count in feature_value_count_dict.items():
            feature_value_data = train_data[train_data[feature_name] == feature_value]
            assigned_to_node = False
            for c in class_list: 
                class_count = feature_value_data[feature_value_data[label] == c].shape[0]
                if class_count == count:
                    tree[feature_value] = c
                    train_data = train_data[train_data[feature_name] != feature_value]
                    assigned_to_node = True
            if not assigned_to_node:
                tree[feature_value] = "?" 
                
        return tree, train_data
    else:
        left_split = train_data[train_data[feature_name] <= split_value]
        right_split = train_data[train_data[feature_name] > split_value]
        tree = {"<=" + str(split_value): {}, ">" + str(split_value): {}}
        
        for data, key in [(left_split, "<=" + str(split_value)), (right_split, ">" + str(split_value))]:
            assigned_to_node = False
            for c in class_list:
                class_count = data[data[label] == c].shape[0]
                if class_count == data.shape[0]:
                    tree[key] = c
                    assigned_to_node = True
            if not assigned_to_node:
                tree[key] = "?"
                
        return tree, train_data

def make_tree(root, prev_feature_value, train_data, label, class_list, processed_values=None):
    if processed_values is None:
        processed_values = set()
    
    if train_data.shape[0] != 0: 
        max_info_feature, split_value = find_most_informative_feature(train_data, label, class_list)
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list, split_value)
        
        if tree:  
            next_root = None
            
            if prev_feature_value is not None:
                root[prev_feature_value] = dict()
                root[prev_feature_value][max_info_feature] = tree
                next_root = root[prev_feature_value][max_info_feature]
            else:
                root[max_info_feature] = tree
                next_root = root[max_info_feature]
            
            if next_root is not None and isinstance(next_root, dict): 
                for node, branch in list(next_root.items()):
                    if branch == "?":
                        feature_value_data = train_data[train_data[max_info_feature] == node]
                        feature_value_hash = hash(feature_value_data.to_string())
                        if feature_value_hash not in processed_values:
                            processed_values.add(feature_value_hash)
                            make_tree(next_root, node, feature_value_data, label, class_list, processed_values)
                    elif isinstance(branch, dict) and "?" in branch.values():
                        missing_data = train_data[train_data[max_info_feature] != node]
                        feature_value_hash = hash(missing_data.to_string())
                        if feature_value_hash not in processed_values:
                            processed_values.add(feature_value_hash)
                            make_tree(next_root, node, missing_data, label, class_list, processed_values)

def prune_tree(tree, validation_data, label, class_list):
    # Реалізація обрізання дерева (необхідно додати власну логіку обрізання)
    pass

def c45(train_data_m, label, prune=False):
    train_data = train_data_m.copy()
    tree = {}
    class_list = train_data[label].unique()
    make_tree(tree, None, train_data, label, class_list)
    if prune:
        prune_tree(tree, train_data, label, class_list)
    return tree

def predictC45(tree, instance):
    if not isinstance(tree, dict):
        return tree 
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predictC45(tree[root_node][feature_value], instance)
        else:
            return None
        
def evaluateС45(tree, test_data_m, label):
    my_list = []
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows(): 
        result = predictC45(tree, test_data_m.iloc[index])
        my_list.append(result)
        if result == test_data_m[label].iloc[index]: 
            correct_preditct += 1 
        else:
            wrong_preditct += 1 
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) 
    print("check prediction: ", my_list)
    return accuracy

'''
# Приклад даних
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Тренування моделі
tree = c45(df, 'PlayTennis')

# Передбачення
instance = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
prediction = predictC45(tree, instance)

print("Tree:", tree)
print("Prediction for instance:", prediction)
'''
#--------------------Employee-(4 000 600)----------------------
train_data_emp = pd.read_csv('emp_data/emp_train_4000.csv')  
test_data_emp = pd.read_csv('emp_data/emp_test_600.csv') 


# ----------------2--C45-----------------------
start_time2 = time.time()  # Початок вимірювання часу
treeC45 = c45(train_data_emp, 'LeaveOrNot')
end_time2 = time.time()  # Кінець вимірювання часу
execution_time2 = end_time2 - start_time2
print(f"Час виконання алгоритму C45: {execution_time2} секунд")
#print(treeC45)
accuracyC45 = evaluateС45(treeC45, test_data_emp, 'LeaveOrNot') 
print("Accuracy for C45 treeeeeee:", accuracyC45)