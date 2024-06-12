import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc

# Генеруємо випадкові дані для класифікації
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Розділяємо дані на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізуємо та навчаємо модель дерева рішень
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Отримуємо ймовірності принадлежності кожного класу на тестовому наборі
probas = model.predict_proba(X_test)

# Рахуємо ROC-криву та обчислюємо площу під нею (AUC)
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
roc_auc = auc(fpr, tpr)

# Виводимо ROC-криву та площу під нею (AUC)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
