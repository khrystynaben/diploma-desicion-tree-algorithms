import pandas as pd
import numpy as np
from collections import defaultdict

# Заданий приклад даних
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Класи та мітки
label = 'PlayTennis'
class_list = ['Yes', 'No']

def entropy(data, label):
    label_counts = data[label].value_counts()
    total_samples = len(data)
    entropy_val = 0
    for cls in class_list:
        if cls in label_counts:
            prob_cls = label_counts[cls] / total_samples
            entropy_val -= prob_cls * np.log2(prob_cls)
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

def cn2(data, label, class_list, min_info_gain=0.1):
    tree = defaultdict(dict)
    best_info_gain = 0

    while len(data) > 0:
        current_entropy = entropy(data, label)
        if best_info_gain == 0 or current_entropy < best_info_gain:
            best_info_gain = current_entropy
            best_attr = None
            best_value = None
            best_subset = None

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

        if best_attr is not None:
            tree[best_attr][best_value] = {}
            data = best_subset
        else:
            break

    return tree




# Функція для генерації нових даних
def generate_new_data(n):
    outlook_values = ['Sunny', 'Overcast', 'Rain']
    temperature_values = np.random.randint(60, 100, n)
    humidity_values = np.random.randint(60, 100, n)
    windy_values = np.random.choice([True, False], n)
    play_tennis_values = np.random.choice(['Yes', 'No'], n)
    
    new_data = pd.DataFrame({
        'Outlook': np.random.choice(outlook_values, n),
        'Temperature': temperature_values,
        'Humidity': humidity_values,
        'Windy': windy_values,
        'PlayTennis': play_tennis_values
    })
    
    return new_data

# Генерація нових даних у 5 разів більшому обсязі
expanded_data = pd.concat([generate_new_data(len(data))] * 5, ignore_index=True)

print("Розширений датасет:")
print(expanded_data)

# Виклик функції для побудови дерева рішень за допомогою алгоритму CN2
decision_tree = cn2(expanded_data, label, class_list)
print(decision_tree)
