import pandas as pd
import numpy as np

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

# Отримуємо унікальні класи та ініціалізуємо матрицю помилок
unique_classes = train_data['PlayTennis'].unique()
conf_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)

# Обчислюємо елементи матриці помилок на основі передбачень для тестового набору
for index, row in test_data.iterrows():
    true_class = row['PlayTennis']
    pred_class = 'Yes' if row['Temperature'] > 75 else 'No'  # Приклад простого правила для передбачення
    conf_matrix[np.where(unique_classes == true_class), np.where(unique_classes == pred_class)] += 1

# Виводимо матрицю помилок
print("Confusion Matrix:")
print(conf_matrix)
