import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Генеруємо змінні та мітки
X = np.random.rand(10, 2)
y = 2*X[:,0] - 3*X[:,1] + 1 + 0.2*np.random.randn(10)

X1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
Y1 = [90,56,80,72,95,100,54,67,72,89,74,59,63,81,93,51,65,72,87,99,52,83,67,54,86,92,56,78,54,95]
# Розділяємо датасет на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Навчаємо класифікатор
clf = LinearRegression()
clf.fit(X_train, y_train)

# Класифікуємо тестові дані та оцінюємо точність
accuracy = clf.score(X_test, y_test)
print(X)
print(y)
print("Accuracy:", accuracy)
'''
import numpy as np
from sklearn.linear_model import LinearRegression

# Задаємо дані
X1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
Y1 = np.array([90,56,80,72,95,100,54,67,72,89,74,59,63,81,93,51,65,72,87,99,52,83,67,54,86,92,56,78,54,95])

# Перетворюємо X1 на двовимірний масив
X1 = X1.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3, random_state=42)
# Створюємо та навчаємо модель лінійної регресії
model = LinearRegression()
model.fit(X_train, y_train)

# Виконуємо передбачення на нових значеннях
X_new = np.array([[31], [32], [33]])  # Нові значення для передбачення
y_pred = model.predict(X_new)

print(y_pred)
# Класифікуємо тестові дані та оцінюємо точність
accuracy = model.score(X_test, y_test)
print("Accuracy for regression:", accuracy)
'''