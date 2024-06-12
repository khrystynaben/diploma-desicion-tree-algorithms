#from _2_tree_c45 import c45, predictС45, evaluateС45
from newww_c45 import new_c45, new_evaluateС45, predict_single
from measure_ROC2 import my_roc_curve
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# Функція для обчислення показника чутливості (True Positive Rate)
def sensitivity(tp, fn):
    return tp / (tp + fn)

# Функція для обчислення правильності (Precision Rate)
def precision(tp, fp):
    return tp / (tp + fp)

# Функція для обчислення специфічності (True Negative Rate)
def specificity(tn, fp):
    return tn / (tn + fp)

# Функція для обчислення F1-оцінки
def f1_score(tp, fp, fn):
    prec = precision(tp, fp)
    sens = sensitivity(tp, fn)
    return 2 * (prec * sens) / (prec + sens)


#--------------------Play Tennis-------------------
train_data1 = pd.read_csv("D:/train/PlayTennis.csv") 
test_data1 = pd.read_csv("D:/test/PlayTennis.csv") 
label1 = 'Play Tennis'

#--------------------Marks-----------------------
train_data2 = pd.read_csv("D:/train/new_tree_data_marks.csv") 
test_data2 = pd.read_csv("D:/test/new_tree_data_marks.csv") 
label2 = 'mark'

#--------------------Divorce---------------------
train_data3 = pd.read_csv('data/divorce/train.csv')
test_data3 = pd.read_csv('data/divorce/test.csv') 
label3 = 'Divorce'

#-------------------Employee-(4 000 600)-------------------
train_data4 = pd.read_csv('data/emp_data/emp_train_4000.csv')  
test_data4 = pd.read_csv('data/emp_data/emp_test_600.csv') 
label4 = 'LeaveOrNot'

#--------------------Satisfaction---------------------
train_data6 = pd.read_csv('data/satisfaction/train20000.csv')  
test_data6 = pd.read_csv('data/satisfaction/test4000.csv') 
label6 = 'satisfaction'

#--------------------Loan---------------------
train_data = pd.read_csv('data/loan!/loan_approval_train_updated.csv')  
test_data = pd.read_csv('data/loan!/loan_approval_test_updated.csv') 
label = 'loan_status'

'''
2303.0799889564514 секунд
ccuracy:  0.9932526294899782
[0 1]
Confusion Matrix:
[[ 160    9]
 [  59 9850]]
'''

X_emp = train_data.drop(columns=[label]).values
y_emp = train_data[label].values
X_emp_test = train_data.drop(columns=[label]).values

model = new_c45(X_emp, y_emp, min_samples_split=2, max_depth=None)


# Функція для передбачення на тестовому наборі
def make_predictions(tree, test_data_m, label):
    predictions = []
    for index, row in test_data_m.iterrows(): 
        result = predict_single(tree, test_data_m.iloc[index])
        if result is None or result == '?':
            result = 0  # або інше значення за замовчуванням
        predictions.append(result)
    return predictions

# Фактичні та передбачені значення
actual_values = test_data[label].values.tolist()
predicted_values = make_predictions(model, test_data, label)

print("actual values: ", actual_values)
print("predicted values: ", predicted_values)


# Перетворення списку на NumPy array
actual_values_array = np.array(actual_values)
predicted_values_array = np.array(predicted_values)

# Перетворення на числовий тип даних
actual_values_numeric = actual_values_array.astype(int)
predicted_values_numeric = predicted_values_array.astype(int) 
# Обчислення показників
tp = np.sum(np.logical_and(predicted_values_numeric == 1, actual_values_numeric == 1))
fp = np.sum(np.logical_and(predicted_values_numeric == 1, actual_values_numeric == 0))
tn = np.sum(np.logical_and(predicted_values_numeric == 0, actual_values_numeric == 0))
fn = np.sum(np.logical_and(predicted_values_numeric == 0, actual_values_numeric == 1))


print("tp:", tp)
print("fp:", fp)
print("tn:", tn)
print("fn:", fn)

# Обчислення показників
se = sensitivity(tp, fn)
pr = precision(tp, fp)
sp = specificity(tn, fp)
f1 = f1_score(tp, fp, fn)

print("Sensitivity (TPR):", se)
print("Precision (PR):", pr)
print("Specificity (SP):", sp)
print("F1 Score:", f1)


actual_values = test_data[label].values.tolist()  
print(make_predictions(model, test_data, label))
print("Accuracy: ", new_evaluateС45(model,test_data,label))

# Отримуємо фактичні та передбачені мітки класів для тестового набору
actual_labels = test_data[label].values
predicted_labels = make_predictions(model, test_data.drop(columns=[label]), label)

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

# Визначаємо кольори для кожної комірки
# Замінимо значення на RGBA послідовності
colors = [
    [0, 1, 0, 0.6],  # Red with 60% opacity
    [1, 0, 0, 0.6],  # Green with 60% opacity
    [1, 0, 0, 0.6],  # Green with 60% opacity
    [0, 1, 0, 0.6]   # Red with 60% opacity
]
cmap = ListedColormap(colors)

# Визначаємо маску для кольорових комірок
color_mask = np.array([
    [0, 1],
    [2, 3]
])

# Відображаємо confusion matrix зі зміненими кольорами
plt.figure(figsize=(6, 6))
plt.imshow(color_mask, interpolation='nearest', cmap=cmap)
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


