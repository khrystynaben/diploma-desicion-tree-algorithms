import numpy as np
import pandas as pd
import time

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

def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
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
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature

def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
    tree = {} 
    
    for feature_value, count in feature_value_count_dict.items():
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        #print(feature_value_data)
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

def make_tree0(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0:
        max_info_feature = find_most_informative_feature(train_data, label, class_list) 
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list)
        
        next_root = None
        
        if prev_feature_value != None:
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else:
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        
        for node, branch in list(next_root.items()):
            if branch == "?":
                feature_value_data = train_data[train_data[max_info_feature] == node]
                make_tree(next_root, node, feature_value_data, label, class_list)
            elif isinstance(branch, dict) and "?" in branch.values():
                missing_data = train_data[train_data[max_info_feature] != node]
                make_tree(next_root, node, missing_data, label, class_list)


def make_tree(root, prev_feature_value, train_data, label, class_list, processed_values=None):
    if processed_values is None:
        processed_values = set()  # Створюємо множину для збереження оброблених значень
    
    if train_data.shape[0] != 0: 
        max_info_feature = find_most_informative_feature(train_data, label, class_list) 
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list)
        
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
                        if feature_value_hash not in processed_values:  # Перевірка на наявність обробленого значення
                            processed_values.add(feature_value_hash)  # Додаємо хеш унікального значення
                            make_tree(next_root, node, feature_value_data, label, class_list, processed_values)
                    elif isinstance(branch, dict) and "?" in branch.values():
                        missing_data = train_data[train_data[max_info_feature] != node]
                        feature_value_hash = hash(missing_data.to_string())
                        if feature_value_hash not in processed_values:  # Перевірка на наявність обробленого значення
                            processed_values.add(feature_value_hash)  # Додаємо хеш унікального значення
                            make_tree(next_root, node, missing_data, label, class_list, processed_values)



def c45(train_data_m, label):
    train_data = train_data_m.copy() 
    tree = {} 
    class_list = train_data[label].unique() 
    make_tree(tree, None, train_data, label, class_list) 
    return tree

def predictС45(tree, instance):
    if not isinstance(tree, dict): #якщо листок
        return tree 
    else:
        root_node = next(iter(tree)) #кореневий елемент
        feature_value = instance[root_node] #значення
        if feature_value in tree[root_node]: #якщо значення існує
            return predictС45(tree[root_node][feature_value], instance) #передбачення для наступного елемента
        else:
            return None

def evaluateС45(tree, test_data_m, label):
    my_list = []
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows(): 
        result = predictС45(tree, test_data_m.iloc[index])
        my_list.append(result)
        if result == test_data_m[label].iloc[index]: 
            correct_preditct += 1 
        else:
            wrong_preditct += 1 
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) 
    print("check prediction: ", my_list)
    return accuracy


# Example usage:
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

'''
tree = c45(data, 'PlayTennis')
print(tree)


accuracy = evaluate(tree, data, 'PlayTennis') #need to add test data
print("Accuracy for tree:", accuracy)
'''

#--------------------Marks-(115 15)----------------------
train_data_marks = pd.read_csv("D:/train/new_tree_data_marks.csv") 
test_data_marks = pd.read_csv("D:/test/new_tree_data_marks.csv")

# ----------------2--C45-----------------------
start_time2 = time.time()  # Початок вимірювання часу
treeC45 = c45(train_data_marks, 'mark')
end_time2 = time.time()  # Кінець вимірювання часу
execution_time2 = end_time2 - start_time2
print(f"Час виконання алгоритму C45: {execution_time2} секунд")

accuracyC45 = evaluateС45(treeC45, test_data_marks, 'mark') 
print(treeC45)
print("Accuracy for C45 treeeeeee:", accuracyC45)