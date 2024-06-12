import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Генеруємо випадкові дані для класифікації
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Розділяємо дані на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізуємо та навчаємо модель дерева рішень
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Зробимо прогнози на тестовому наборі
y_pred = model.predict(X_test)


cm = confusion_matrix(y_test, y_pred)

# ----------------------------------Обчислюємо метрики ефективності

#--------1
#точність
#AC = (TP+TN)/(TP+TN+FP+FN)
accuracy = accuracy_score(y_test, y_pred) 


#--------2
#Коефіцієнт неправильної класифікації 
#MR Misclassification Rate 
#MR = (FP+FN)/(TP+TN+FP+FN) або (1-AC)


#--------3
#sensitivity 
#чутливість
#SE=TPR= TP/(TP+FN)
recall = recall_score(y_test, y_pred)


#--------4
#правильність
#PR = TP/(TP+FP)
precision = precision_score(y_test, y_pred) 


#--------5
# Хибний позитивний коефіцієнт 
#FRP
#FPR = FP/(TN+FP)


#--------6
#specifity
# спечифічність
#SP =TNR=TN/(TN+FP)



#--------7
# ф1 - міра
f1 = f1_score(y_test, y_pred)


# Виведемо метрики ефективності
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)

# Візуалізуємо confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
