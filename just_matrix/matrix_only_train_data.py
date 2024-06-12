import pandas as pd
import numpy as np

# Створюємо DataFrame з даними
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Отримуємо унікальні класи та ініціалізуємо матрицю помилок
unique_classes = data['PlayTennis'].unique()
conf_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)

# Обчислюємо елементи матриці помилок
for index, row in data.iterrows():
    true_class = row['PlayTennis']
    pred_class = true_class  # У цьому прикладі передбаченням є фактичний клас
    conf_matrix[np.where(unique_classes == true_class), np.where(unique_classes == pred_class)] += 1

# Виводимо матрицю помилок
print("Confusion Matrix:")
print(conf_matrix)
