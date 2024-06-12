import random
import math
import pandas as pd
from collections import Counter
'''
# Приклад даних
data = [
    ['A', 'A', 'A', 5, 'at_yes', 'yes'],
    ['C', 'C', 'A', 3, 'at_yes', 'no'],
    ['D', 'A', 'D', 1, 'at_yes', 'no'],
    ['B', 'C', 'A', 6, 'at_yes', 'yes']
]
'''
# Клас для вузла дерева рішень
class Node:
    def __init__(self, attribute=None, threshold=None, left=None, right=None, result=None):
        self.attribute = attribute  # атрибут, по якому проводиться розбиття
        self.threshold = threshold  # порог розбиття для числових атрибутів
        self.left = left  # лівий піддерево
        self.right = right  # правий піддерево
        self.result = result  # результат для листового вузла

# Реалізація алгоритму дерева рішень
# for marks
def decision_tree_learning0(data):
    # Якщо всі приклади мають однаковий результат, створюємо листовий вузол з цим результатом
    results = [example[-1] for example in data]
    if len(set(results)) == 1:
        return Node(result=results[0])

    # Вибираємо найкращий атрибут та порог для розбиття
    best_attr, best_threshold = select_best_attribute(data)
    left_data, right_data = split_data(data, best_attr, best_threshold)

    # Рекурсивно будуємо ліве та праве піддерево
    left_subtree = decision_tree_learning(left_data)
    right_subtree = decision_tree_learning(right_data)

    # Повертаємо вузол з розбиттям
    return Node(attribute=best_attr, threshold=best_threshold, left=left_subtree, right=right_subtree)

def decision_tree_learning(data):
    # Якщо всі приклади мають однаковий результат, створюємо листовий вузол з цим результатом
    results = [example[-1] for example in data]
    if len(set(results)) == 1:
        return Node(result=results[0])

    # Вибираємо найкращий атрибут та порог для розбиття
    best_attr, best_threshold = select_best_attribute(data)
    if best_attr is None or best_threshold is None:
        return Node(result=Counter(results).most_common(1)[0][0])  # обробка ситуації, коли не можна зробити розбиття

    left_data, right_data = split_data(data, best_attr, best_threshold)
    if not left_data or not right_data:  # перевірка на порожні підмножини даних
        return Node(result=Counter(results).most_common(1)[0][0])  # обробка ситуації, коли не можна зробити розбиття

    # Рекурсивно будуємо ліве та праве піддерево
    left_subtree = decision_tree_learning(left_data)
    right_subtree = decision_tree_learning(right_data)

    # Повертаємо вузол з розбиттям
    return Node(attribute=best_attr, threshold=best_threshold, left=left_subtree, right=right_subtree)



# Функція для вибору найкращого атрибуту та порогу для розбиття
def select_best_attribute(data):
    best_attr = None
    best_threshold = None
    best_gain = 0

    for attr_index in range(len(data[0]) - 1):  # ітеруємося по всіх атрибутах крім останнього (результату)
        # Якщо атрибут є числовим, використовуємо його значення як порог розбиття
        if isinstance(data[0][attr_index], int):
            values = sorted(set([example[attr_index] for example in data]))
            thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        else:
            thresholds = set([example[attr_index] for example in data])

        for threshold in thresholds:
            gain = information_gain(data, attr_index, threshold)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr_index
                best_threshold = threshold

    return best_attr, best_threshold

# Функція для обчислення приросту інформації (Information Gain)
def information_gain(data, attr_index, threshold):
    left_data, right_data = split_data(data, attr_index, threshold)
    p_left = len(left_data) / len(data)
    p_right = len(right_data) / len(data)
    return entropy(data) - (p_left * entropy(left_data) + p_right * entropy(right_data))

# Функція для обчислення ентропії
def entropy(data):
    results = [example[-1] for example in data]
    counter = Counter(results)
    probabilities = [count / len(results) for count in counter.values()]
    return -sum(p * math.log2(p) for p in probabilities)

# Функція для розбиття даних за певним атрибутом та порогом
def split_data(data, attr_index, threshold):
    left_data = []
    right_data = []
    for example in data:
        if isinstance(example[attr_index], int):
            if example[attr_index] <= threshold:
                left_data.append(example)
            else:
                right_data.append(example)
        else:
            if example[attr_index] == threshold:
                left_data.append(example)
            else:
                right_data.append(example)
    return left_data, right_data

# Функція для прогнозування за допомогою дерева рішень
def predict(tree, example):
    if tree.result is not None:
        return tree.result

    if isinstance(example[tree.attribute], int):
        if example[tree.attribute] <= tree.threshold:
            return predict(tree.left, example)
        else:
            return predict(tree.right, example)
    else:
        if example[tree.attribute] == tree.threshold:
            return predict(tree.left, example)
        else:
            return predict(tree.right, example)

# Створюємо випадковий ліс
def random_forest_learning(data, num_trees):
    trees = []
    for _ in range(num_trees):
        # Генеруємо підмножину даних для кожного дерева
        subset = [random.choice(data) for _ in range(len(data))]
        tree = decision_tree_learning(subset)
        trees.append(tree)
    return trees

# Прогнозуємо за допомогою випадкового лісу
def random_forest_predict(trees, example):
    predictions = [predict(tree, example) for tree in trees]
    return Counter(predictions).most_common(1)[0][0]

def accuracy_new_forest(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) 


'''
train_data_marks = pd.read_csv("D:/train/tree_data_marks.csv") 
test_data_marks = pd.read_csv("D:/test/tree_data_marks.csv") 
data = train_data_marks.values.tolist()
test_data = test_data_marks.values.tolist()


# Прогнозуємо для кожного прикладу
predictions = []
trees = random_forest_learning(data, 3)
for example in test_data:
    prediction = random_forest_predict(trees, example)
    predictions.append(prediction)
    #print(f"Example: {example}, Prediction: {prediction}")

print('Prediction:', predictions)

actual_values = test_data_marks['mark'].values.tolist()  # Справжні значення міток
accuracy = accuracy_new_forest(actual_values, predictions)  # predictions - передбачені значення
print('Accuracy:', accuracy)
'''