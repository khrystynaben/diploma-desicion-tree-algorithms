import numpy as np
import pandas as pd

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
'''
def generate_rules(train_data, label, class_list):
    rules = []
    remaining_data = train_data.copy()
    print('here4')
    while remaining_data.shape[0] != 0:
        print(remaining_data.shape[0])
        max_info_feature = find_most_informative_feature(remaining_data, label, class_list)
        feature_value_list = remaining_data[max_info_feature].unique()
        print('here5')
        for feature_value in feature_value_list:
            print(feature_value_list)
            rule_condition = f"{max_info_feature} == '{feature_value}'"
            rule_data = remaining_data.query(rule_condition)
            rule_class_counts = rule_data[label].value_counts()
            print('here6')
            if len(rule_class_counts) == 1: # Єдиний клас в правилі
                print('here6')
                rule_class = rule_class_counts.index[0]
                rule = f"If {rule_condition} then {label} is {rule_class}"
                rules.append(rule)
                remaining_data = remaining_data.query(f"{max_info_feature} != '{feature_value}'")
    print('here7')
    return rules

'''
def generate_rules(train_data, label, class_list):
    rules = []
    remaining_data = train_data.copy()
    
    while remaining_data.shape[0] != 0:
        max_info_feature = find_most_informative_feature(remaining_data, label, class_list)
        feature_value_list = remaining_data[max_info_feature].unique()
        
        found_rule = False
        for feature_value in feature_value_list:
            rule_condition = f"{max_info_feature} == '{feature_value}'"
            rule_data = remaining_data.query(rule_condition)
            rule_class_counts = rule_data[label].value_counts()
            
            if len(rule_class_counts) == 1:  # Єдиний клас в правилі
                rule_class = rule_class_counts.index[0]
                rule = f"If {rule_condition} then {label} is {rule_class}"
                rules.append(rule)
                remaining_data = remaining_data[remaining_data[max_info_feature] != feature_value]  # Видаляємо записи зі значенням feature_value
                found_rule = True
        
        if not found_rule:  # Якщо не знайдено правило, додати правило для непокритих даних
            majority_class = remaining_data[label].value_counts().idxmax()
            rule = f"If {max_info_feature} is anything then {label} is {majority_class}"
            rules.append(rule)
            break
    
    return rules

'''
def generate_rules(train_data, label, class_list):
    rules = []
    remaining_data = train_data.copy()
    
    while remaining_data.shape[0] != 0:
        max_info_feature = find_most_informative_feature(remaining_data, label, class_list)
        feature_value_list = remaining_data[max_info_feature].unique()
        
        found_rule = False
        for feature_value in feature_value_list:
            rule_condition = f"{max_info_feature} == '{feature_value}'"
            rule_data = remaining_data.query(rule_condition)
            rule_class_counts = rule_data[label].value_counts()
            print(len(rule_class_counts))
            if len(rule_class_counts) == 1: # Єдиний клас в правилі
                rule_class = rule_class_counts.index[0]
                rule = f"If {rule_condition} then {label} is {rule_class}"
                rules.append(rule)
                remaining_data = remaining_data.query(f"{max_info_feature} != '{feature_value}'")
                found_rule = True
        
        if not found_rule:  # Якщо не знайдено правило, додати правило для непокритих даних
            majority_class = remaining_data[label].value_counts().idxmax()
            rule = f"If {max_info_feature} is anything then {label} is {majority_class}"
            rules.append(rule)
            break
    
    return rules
'''

# Приклад використання:
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})
print('here')
rules = generate_rules(data, 'PlayTennis', ['Yes', 'No'])
print('here3')
for rule in rules:
    print('here2')
    print(rule)
