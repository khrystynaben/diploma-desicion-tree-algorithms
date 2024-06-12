import pandas as pd
import numpy as np


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

# працює для тенісу та оцінок
def chaid_tree0(data, label, class_list):
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
    
def chaid_tree(data, label, class_list):
    if len(data[label].unique()) == 1:
        return {'prediction': data[label].iloc[0]}
    elif len(data.columns) == 1:
        return {'prediction': data[label].mode().iloc[0]}
    else:
        best_feature, best_split_value = best_split_chi_square(data, label, class_list)
        if best_feature is None or best_split_value is None:
            return {'prediction': data[label].mode().iloc[0]}  # Якщо немає належного розділення, повертаємо моду
        data_below = data[data[best_feature] == best_split_value]
        data_above = data[data[best_feature] != best_split_value]
        subtree = {'feature': best_feature, 'split_value': best_split_value}
        if len(data_below) == 0 or len(data_above) == 0:
            subtree['prediction'] = data[label].mode().iloc[0]
        else:
            subtree['below'] = chaid_tree(data_below, label, class_list)
            subtree['above'] = chaid_tree(data_above, label, class_list)
        return subtree

#not working
def predict1(tree, instance):
    if not isinstance(tree, dict):  # Якщо не листок, повертаємо значення
        return tree
    else:
        root_node = next(iter(tree))  # Отримуємо кореневий вузол
        print("!!!!!!", root_node)
        feature_value = instance[root_node]  # Отримуємо значення функції для даного вузла
        if feature_value in tree[root_node]:  # Перевіряємо, чи існує значення в дереві
            return predict1(tree[root_node][feature_value], instance)  # Рекурсивно переходимо до наступного вузла
        else:
            return None  # Якщо значення не знайдено, повертаємо None
#return None
def predict2(tree, instance):
    if not isinstance(tree, dict):  # Якщо не листок, повертаємо значення
        return tree
    else:
        root_node = next(iter(tree), None)  # Отримуємо кореневий вузол
        if root_node is None or root_node not in instance:
            return None  # Якщо вузол не знайдено або не існує у даних, повертаємо None
        feature_value = instance[root_node]  # Отримуємо значення функції для даного вузла
        if feature_value in tree[root_node]:  # Перевіряємо, чи існує значення в дереві
            return predict2(tree[root_node][feature_value], instance)  # Рекурсивно переходимо до наступного вузла
        else:
            return None  # Якщо значення не знайдено, повертаємо None

def predict(tree, instance):
    if 'prediction' in tree:  # Якщо це листок, повертаємо прогноз
        return tree['prediction']
    else:
        feature = tree['feature']
        split_value = tree['split_value']
        instance_value = instance.get(feature)
        #print(instance_value)
        if instance_value is None:
            return None  # Якщо відсутнє значення функції для вузла, повертаємо None
        if instance_value == split_value:
            return predict(tree['below'], instance)  # Рекурсивно переходимо до "нижнього" піддерева
        else:
            return predict(tree['above'], instance)  # Рекурсивно переходимо до "верхнього" піддерева

#works
def evaluateChaid(tree, test_data, label):
    correct_predict = 0
    for index, row in test_data.iterrows():
        result = predict(tree, row)
        #print("result ",result, " = row[label] ",row[label])
        if result == row[label]:
            correct_predict += 1
    accuracy = correct_predict / len(test_data)
    return accuracy

#also works
def evaluateChaid2(tree, test_data, label):
    correct_predict = 0
    total_instances = len(test_data)
    for index, row in test_data.iterrows():
        result = predict(tree, row)
        #print("result ", result)
        #print("row[label] ", row[label])
        if result == row[label]:
            correct_predict += 1
    accuracy = correct_predict / total_instances if total_instances > 0 else 0
    return accuracy


# Задані дані з грою в теніс
data0 = pd.DataFrame({
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
#------------------------Play Tennis---------------------------------
train_data_tennis = pd.read_csv("D:/train/PlayTennis.csv") 
test_data_tennis = pd.read_csv("D:/test/PlayTennis.csv") 
label = 'Play Tennis'

# Побудова дерева рішень CHAID
decision_tree = chaid_tree(train_data_tennis, label, class_list)
print(decision_tree)

accuracyChaid = evaluateChaid(decision_tree, test_data_tennis, 'Play Tennis') 
print("Accuracy for Chaid treeeeeee:", accuracyChaid)
'''
'''
#0.53
#--------------------------marks-----------------------------------------
train_data_marks = pd.read_csv("D:/train/new_tree_data_marks.csv") 
test_data_marks = pd.read_csv("D:/test/new_tree_data_marks.csv") 
class_list = ['yes', 'no']

# Побудова дерева рішень CHAID
decision_tree = chaid_tree(train_data_marks, 'mark', class_list)
print(decision_tree)

accuracyChaid = evaluateChaid(decision_tree, test_data_marks, 'mark') 
print("Accuracy for Chaid tree:", accuracyChaid)
'''
'''
#0.66 (all zeros)
train_data_emp = pd.read_csv('emp_data/emp_train_4000.csv')  
test_data_emp = pd.read_csv('emp_data/emp_test_600.csv')

label = 'LeaveOrNot'
class_list = ['0', '1']

treeChaid = chaid_tree(train_data_emp,  label, class_list)
accuracyChaid = evaluateChaid(treeChaid, test_data_emp, 'LeaveOrNot') 
print("Accuracy for Chaid treeeeeee:", accuracyChaid)
'''