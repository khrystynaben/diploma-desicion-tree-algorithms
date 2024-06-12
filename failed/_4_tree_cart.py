import numpy as np
import pandas as pd
import sys

def calc_gini_index(train_data, label, class_list):
    total_row = train_data.shape[0]
    total_gini = 0
    
    for c in class_list:
        total_class_count = train_data[train_data[label] == c].shape[0]
        total_class_gini = (total_class_count/total_row) ** 2
        total_gini += total_class_gini 
    print(total_gini)
    return 1 - total_gini

def calc_gini_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_gini_gain = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_gini = calc_gini_index(feature_value_data, label, class_list) 
        feature_value_probability = feature_value_count/total_row
        feature_gini_gain += feature_value_probability * feature_value_gini
        
    return calc_gini_index(train_data, label, class_list) - feature_gini_gain

def find_best_split(train_data, label, class_list):
    feature_list = train_data.columns.drop(label)
    
    max_gini_gain = -1
    best_split_feature = None
    
    for feature in feature_list:  
        feature_gini_gain = calc_gini_gain(feature, train_data, label, class_list)
        print("gini", best_split_feature)
        if max_gini_gain < feature_gini_gain:
            max_gini_gain = feature_gini_gain
            best_split_feature = feature
            
    return best_split_feature

def generate_sub_tree(feature_name, train_data, label, class_list):
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

def cart(train_data, label, class_list):
    tree = {} 
    make_sub_tree(tree, None, train_data, label, class_list) 
    return tree

def make_sub_tree(root, prev_feature_value, train_data, label, class_list,max_depth=sys.getrecursionlimit()):
    if train_data.shape[0] != 0 and max_depth > 0:
        best_split_feature = find_best_split(train_data, label, class_list) 
        sub_tree, train_data = generate_sub_tree(best_split_feature, train_data, label, class_list)
        next_root = None
        
        if prev_feature_value != None:
            root[prev_feature_value] = dict()
            root[prev_feature_value][best_split_feature] = sub_tree
            next_root = root[prev_feature_value][best_split_feature]
        else:
            root[best_split_feature] = sub_tree
            next_root = root[best_split_feature]
        
        for node, branch in list(next_root.items()):
            if branch == "?":
                print("best split", best_split_feature)
                feature_value_data = train_data[train_data[best_split_feature] == node]
                print("FEATURE: ",feature_value_data)
                make_sub_tree(next_root, node, feature_value_data, label, class_list)

'''
def predict(tree, instance):
    if not isinstance(tree, dict):  # Якщо не листок, повертаємо значення
        return tree
    else:
        root_node = next(iter(tree))  # Отримуємо кореневий вузол
        feature_value = instance[root_node]  # Отримуємо значення функції для даного вузла
        if feature_value in tree[root_node]:  # Перевіряємо, чи існує значення в дереві
            return predict(tree[root_node][feature_value], instance)  # Рекурсивно переходимо до наступного вузла
        else:
            return None  # Якщо значення не знайдено, повертаємо None


def evaluateCart(tree, test_data, label):
    correct_predict = 0
    for index, row in test_data.iterrows():
        result = predict(tree, row)
        if result == row[label]:
            correct_predict += 1
    accuracy = correct_predict / len(test_data)
    return accuracy
'''
def predict(tree, instance):
    if not isinstance(tree, dict):  # Якщо не листок, повертаємо значення
        return tree
    else:
        root_node = next(iter(tree))  # Отримуємо кореневий вузол
        feature_value = instance[root_node]  # Отримуємо значення функції для даного вузла
        if feature_value in tree[root_node]:  # Перевіряємо, чи існує значення в дереві
            return predict(tree[root_node][feature_value], instance)  # Рекурсивно переходимо до наступного вузла
        else:
            return None  # Якщо значення не знайдено, повертаємо None

def evaluateCart(tree, test_data, label):
    correct_predict = 0
    for index, row in test_data.iterrows():
        result = predict(tree, row)
        if result == row[label]:
            correct_predict += 1
    accuracy = correct_predict / len(test_data)
    return accuracy

# Example usage:
# train_data = your_training_data
# label = 'target_variable_column_name'
# class_list = [list_of_classes]
# tree = cart(train_data, label, class_list)


# Example data
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Example data
test_data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast'],
    'Temperature': [86, 80, 83],
    'Humidity': [85, 91, 78],
    'Windy': [False, True, False],
    'PlayTennis': ['No', 'No', 'Yes']
})

'''
# Example usage of the build_tree function
label = 'PlayTennis'
class_list = ['Yes', 'No']
tree = cart(data, label, class_list)

# Print the built tree
print(tree)

# Припустимо, що у вас є побудоване дерево tree і тестові дані test_data з міткою 'PlayTennis'
accuracy = evaluateCart(tree, test_data, 'PlayTennis')
print(f"Accuracy: {accuracy}")
'''