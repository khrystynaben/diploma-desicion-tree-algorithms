import pandas as pd
from collections import defaultdict
import math

# Задані дані з грою в теніс
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Задані класи та мітки
label = 'PlayTennis'
class_list = ['Yes', 'No']

def entropy(data, label):
    # Обчислення ентропії для заданого набору даних і мітки класу
    label_counts = data[label].value_counts()
    total_samples = len(data)
    entropy_val = 0
    for cls in class_list:
        if cls in label_counts:
            prob_cls = label_counts[cls] / total_samples
            entropy_val -= prob_cls * math.log(prob_cls, 2)
    return entropy_val

def cn2(data, label, class_list):
    # Створення пустого дерева рішень
    tree = defaultdict(dict)
    # Ініціалізація нульової ентропії
    best_entropy = 0

    while len(data) > 0:
        # Обчислення поточної ентропії
        current_entropy = entropy(data, label)
        # Якщо це перша ітерація або зменшилась ентропія, оновити значення
        if best_entropy == 0 or current_entropy < best_entropy:
            best_entropy = current_entropy
            best_attr = None
            best_value = None
            best_subset = None

        # Пройтися по всім атрибутам крім мітки
        for attr in data.columns:
            if attr != label:
                values = data[attr].unique()
                for value in values:
                    # Розбити дані на дві підмножини: одну з поточним значенням атрибута, іншу - без нього
                    subset_with_attr_value = data[data[attr] == value]
                    subset_without_attr_value = data[data[attr] != value]
                    # Обчислити виграш інформації
                    info_gain = current_entropy - (len(subset_with_attr_value) / len(data) * entropy(subset_with_attr_value, label)
                                                    + len(subset_without_attr_value) / len(data) * entropy(subset_without_attr_value, label))
                    # Якщо виграш інформації кращий за попередній, оновити значення
                    if info_gain > best_entropy:
                        best_entropy = info_gain
                        best_attr = attr
                        best_value = value
                        best_subset = subset_with_attr_value

        # Додати нове рішення в дерево
        if best_attr is not None:
            tree[best_attr][best_value] = {}
            data = best_subset
        else:
            break

    return tree

decision_tree = cn2(data, label, class_list)
print(decision_tree)
