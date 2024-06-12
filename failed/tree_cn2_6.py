import pandas as pd
import numpy as np
from collections import defaultdict

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

# Функція для обчислення ентропії
def entropy(data, label):
    label_counts = data[label].value_counts()
    total_samples = len(data)
    entropy_val = 0
    for cls in label_counts.index:
        prob_cls = label_counts[cls] / total_samples
        entropy_val -= prob_cls * np.log2(prob_cls)
    return entropy_val

# Функція для обчислення виграшу інформації
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

# Алгоритм CN2 з додатковими умовами
def cn2(data, label, class_dict, min_info_gain=0.05, min_samples_split=2):
    tree = defaultdict(dict)
    best_info_gain = 0

    if len(data) >= min_samples_split:
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
                    if info_gain > min_info_gain and info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_attr = attr
                        best_value = value
                        best_subset = subset_with_attr_value

        if best_attr is not None:
            tree[best_attr][best_value] = cn2(best_subset, label, class_dict, min_info_gain, min_samples_split)
        else:
            return max(class_dict, key=class_dict.get)  # Повертаємо клас з найбільшою кількістю

    return tree

# Заданий початковий датасет
#data = generate_new_data(10000)

np.random.seed(0)  # Для відтворюваності результатів
n_samples = 1000
age = np.random.randint(18, 65, n_samples)
income_levels = np.random.choice(['Low', 'Medium', 'High'], n_samples)
education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
employment_status = np.random.choice(['Unemployed', 'Part-Time', 'Full-Time'], n_samples)
purchase = np.random.choice(['Yes', 'No'], n_samples)

data = pd.DataFrame({
    'Age': age,
    'Income': income_levels,
    'Education': education,
    'Employment': employment_status,
    'Purchase': purchase
})

print(data)

# Визначення класів та їх кількостей
class_counts = data['Purchase'].value_counts().to_dict()

# Виклик функції для побудови дерева рішень за допомогою алгоритму CN2 з додатковими умовами
decision_tree = cn2(data, 'Purchase', class_counts, min_info_gain=0.05, min_samples_split=2)
print("Дерево рішень:")
print(decision_tree)
