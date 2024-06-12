from _4_new_cart import new_cart, accuracy_new_cart, predict
from sklearn.metrics import roc_curve, auc
from measure_ROC2 import my_roc_curve
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
'''
# Створюємо тренувальний та тестовий набори даних
train_data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

test_data = pd.DataFrame({
    'Outlook': ['Sunny', 'Overcast', 'Rain', 'Sunny'],
    'Temperature': [70, 82, 75, 78],
    'Humidity': [90, 70, 80, 85],
    'Windy': [False, True, False, True],
    'PlayTennis': ['Yes', 'Yes', 'No', 'No']  # Фактичні класи для тестового набору
})
label = 'PlayTennis'
'''
train_data1 = pd.read_csv("D:/train/PlayTennis.csv") 
test_data1 = pd.read_csv("D:/test/PlayTennis.csv") 
label1 = 'Play Tennis'

train_data = pd.read_csv("D:/train/new_tree_data_marks.csv") 
test_data = pd.read_csv("D:/test/new_tree_data_marks.csv") 
label = 'mark'

#--------------------Employee-(4 000 600)----------------------
train_data3 = pd.read_csv('emp_data/emp_train_4000.csv')  
test_data3 = pd.read_csv('emp_data/emp_test_600.csv') 
label3 = 'LeaveOrNot'

# Навчаємо модель на тренувальному наборі
max_depth = 100  # Максимальна глибина дерева
min_size = 1  # Мінімальна кількість зразків для поділу вузла
model = new_cart(train_data.values, max_depth, min_size)


# Функція для передбачення на тестовому наборі
def make_predictions(model, test_data):
    predictions = []
    for row in test_data.values:
        prediction = predict(model, row)
        predictions.append(prediction)
    return predictions

actual_values = test_data[label].values.tolist()  
print(make_predictions(model, test_data))
print("Accuracy: ", accuracy_new_cart(actual_values, make_predictions(model, test_data)))

# Отримуємо фактичні та передбачені мітки класів для тестового набору
actual_labels = test_data[label].values
predicted_labels = make_predictions(model, test_data.drop(columns=[label]))

# Обчислюємо матрицю помилок
unique_classes = np.unique(actual_labels)
print(unique_classes)
conf_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
for true_class, pred_class in zip(actual_labels, predicted_labels):
    true_index = np.where(unique_classes == true_class)[0][0]
    pred_index = np.where(unique_classes == pred_class)[0][0]
    conf_matrix[true_index, pred_index] += 1

# Виводимо матрицю помилок
print("Confusion Matrix:")
print(conf_matrix)

# Визначаємо кольори для кожного типу значення в матриці помилок
colors = ['red', 'green']
cmap = ListedColormap(colors)

# Відображаємо confusion matrix зі зміненими кольорами
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)

plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
tick_marks = np.arange(len(unique_classes))
plt.xticks(tick_marks, unique_classes)
plt.yticks(tick_marks, unique_classes)

num_classes = len(unique_classes)

# Додаємо значення до кожної клітинки у матриці
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(conf_matrix[i, j], '.0f'), horizontalalignment="center", color="black")


plt.show()



my_roc_curve(actual_labels,predicted_labels)

