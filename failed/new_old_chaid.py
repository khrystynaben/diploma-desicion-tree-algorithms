#not works with big data

import pandas as pd
import numpy as np

# Задані дані з грою в теніс
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Задані класи та мітки
label0 = 'PlayTennis'
class_list0 = ['Yes', 'No']
'''
label = 'mark'
class_list = ['yes', 'no']

train_data_marks = pd.read_csv("D:/train/new_tree_data_marks.csv") 
test_data_marks = pd.read_csv("D:/test/new_tree_data_marks.csv") 
'''
train_data_emp = pd.read_csv('data/emp_data/emp_train_4000.csv')  
test_data_emp = pd.read_csv('data/emp_data/emp_test_600.csv')

label = 'LeaveOrNot'
class_list = ['0', '1']
# Функція для обчислення Хі-квадрат для категоріальних змінних
def chi_square(observed, expected):
    return np.sum((observed - expected)**2 / expected)

# Функція для розрахунку значень Хі-квадрат для всіх можливих розділень
def chi_square_split(data, label, class_list, feature):
    splits = data[feature].unique()
    chi_square_values = []
    for split in splits:
        observed = []
        expected = []
        for cls in class_list:
            observed_count = len(data[(data[feature] == split) & (data[label] == cls)])
            observed.append(observed_count)
            expected_count = len(data[data[feature] == split]) * len(data[data[label] == cls]) / len(data)
            expected.append(expected_count)
        chi_square_values.append(chi_square(np.array(observed), np.array(expected)))
    return splits, chi_square_values

# Функція для вибору найкращого розділення на основі Хі-квадрат
def best_split_chi_square(data, label, class_list):
    best_feature = None
    best_split_value = None
    best_chi_square = np.inf
    for feature in data.columns:
        if feature != label:
            splits, chi_square_values = chi_square_split(data, label, class_list, feature)
            min_chi_square_index = np.argmin(chi_square_values)
            min_chi_square = chi_square_values[min_chi_square_index]
            if min_chi_square < best_chi_square:
                best_chi_square = min_chi_square
                best_feature = feature
                best_split_value = splits[min_chi_square_index]
    return best_feature, best_split_value

# Функція для побудови дерева рішень CHAID
def chaid_tree(data, label, class_list):
    if len(data[label].unique()) == 1:
        return {'prediction': data[label].iloc[0]}
    elif len(data.columns) == 1:
        return {'prediction': data[label].mode().iloc[0]}
    else:
        best_feature, best_split_value = best_split_chi_square(data, label, class_list)
        data_below = data[data[best_feature] == best_split_value]
        data_above = data[data[best_feature] != best_split_value]
        subtree = {'feature': best_feature, 'split_value': best_split_value}
        if len(data_below) == 0 or len(data_above) == 0:
            subtree['prediction'] = data[label].mode().iloc[0]
        else:
            subtree['below'] = chaid_tree(data_below, label, class_list)
            subtree['above'] = chaid_tree(data_above, label, class_list)
        return subtree

# Функція для прогнозування на основі побудованого дерева
def predict(tree, sample):
    if 'prediction' in tree:
        return tree['prediction']
    else:
        if sample[tree['feature']] == tree['split_value']:
            return predict(tree['below'], sample)
        else:
            return predict(tree['above'], sample)

# Функція для оцінки точності на тестовій множині
def evaluate_accuracy(tree, test_data):
    correct_predictions = 0
    for idx, sample in test_data.iterrows():
        prediction = predict(tree, sample)
        print(prediction, "==", sample[label])
        if prediction == sample[label]:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_data)
    return accuracy

# Розділення даних на навчальну та тестову множини
train_data = data.iloc[:8]
test_data = data.iloc[8:]



# Побудова дерева рішень CHAID на навчальних даних
decision_tree = chaid_tree(train_data_emp, label, class_list)

# Оцінка точності на тестовій множині
accuracy = evaluate_accuracy(decision_tree, test_data_emp)
print(f"Accuracy: {accuracy}")
