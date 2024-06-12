import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Задаємо фактичні та передбачені мітки класів
actual_labels = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
predicted_labels = np.array([1, 1, 1, 1, 0, 1, 0, 0, 1, 1])

# Знаходимо розмір матриці та ініціалізуємо confusion matrix
unique_classes = np.unique(actual_labels)
num_classes = len(unique_classes)
conf_matrix = np.zeros((num_classes, num_classes))

# Обчислюємо елементи confusion matrix
for i in range(len(actual_labels)):
    true_class = actual_labels[i]
    pred_class = predicted_labels[i]
    conf_matrix[true_class, pred_class] += 1


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

# Додаємо значення до кожної клітинки у матриці
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(conf_matrix[i, j], '.0f'), horizontalalignment="center", color="black")


plt.show()


'''
# Відображаємо confusion matrix графічно
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)


plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(unique_classes))
plt.xticks(tick_marks, unique_classes)
plt.yticks(tick_marks, unique_classes)
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')

# Додаємо значення до кожної клітинки у матриці
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(conf_matrix[i, j], '.0f'), horizontalalignment="center", color="black")


plt.show()
'''