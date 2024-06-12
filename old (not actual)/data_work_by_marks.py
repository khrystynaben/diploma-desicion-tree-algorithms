import numpy as np
import pandas as pd
from _1_tree_id3 import id3, predictID3, evaluateID3
from _2_tree_c45 import c45, evaluateС45
from _3_tree_c50 import c50, evaluateC50
from _4_tree_cart import cart, evaluateCart
from _4_new_cart import new_cart, accuracy_new_cart, predict
from _5_tree_chaid4 import chaid_tree, evaluateChaid
from _6_tree_forest import RandomForest
from _6_new_forest import accuracy_new_forest, random_forest_learning, random_forest_predict
import sys

#sys.setrecursionlimit(30000)


#--------------------Marks-(30 5)----------------------
train_data_marks = pd.read_csv("D:/train/tree_data_marks.csv") 
test_data_marks = pd.read_csv("D:/test/tree_data_marks.csv") 
#print(train_data_m)


# ----------------1--ID3-----------------------
treeID3 = id3(train_data_marks, 'mark')
print(treeID3)
accuracyID3 = evaluateID3(treeID3, test_data_marks, 'mark') 
print("Accuracy for ID3 treeeeeee:", accuracyID3)


# ----------------2--C45-----------------------
treeC45 = c45(train_data_marks, 'mark')
print(treeC45)
accuracyC45 = evaluateС45(treeC45, test_data_marks, 'mark') 
print("Accuracy for C45 treeeeeee:", accuracyC45)


# ----------------4--CART-----------------------
treeCart = new_cart(train_data_marks.values.tolist(), 100, 1)
#print(tree)

predictions = []
for index, row in test_data_marks.iterrows():
    prediction = predict(treeCart, row)
    predictions.append(prediction)

# Приклад використання
actual_values = test_data_marks['mark'].values.tolist()  # Справжні значення міток
accuracy = accuracy_new_cart(actual_values, predictions)  # predictions - передбачені значення
print('Accuracy for Cart:', accuracy)
'''

# -----------------5-CHAID-----------------------
label = 'mark'
class_list = ['Yes', 'No']
treeChaid = chaid_tree(train_data_marks,  label, class_list)
print(treeChaid)
accuracyChaid = evaluateChaid(treeChaid, test_data_marks, 'mark') 
print("Accuracy for Chaid treeeeeee:", accuracyChaid)

'''

# -----------------6-Forest-----------------------

# Прогнозуємо для кожного прикладу
predictions = []
data = train_data_marks.values.tolist()
test_data = test_data_marks.values.tolist()
trees = random_forest_learning(data, 3)
for example in test_data:
    prediction = random_forest_predict(trees, example)
    predictions.append(prediction)
    #print(f"Example: {example}, Prediction: {prediction}")

print('Prediction:', predictions)

actual_values = test_data_marks['mark'].values.tolist()  # Справжні значення міток
accuracy = accuracy_new_forest(actual_values, predictions)  # predictions - передбачені значення
print('Accuracy for Forest:', accuracy)

'''
#результат 0.5
treeForest = RandomForest(n_estimators=50, max_depth=5)
X_train = train_data_marks[['math', 'physics', 'chemestry', 'additional', 'attendence']].values
y_train = train_data_marks['mark'].values
treeForest.fit(X_train,y_train)

X_test = test_data_marks[['math', 'physics', 'chemestry', 'additional', 'attendence']].values
y_test = test_data_marks['mark'].values
accuracyForest = treeForest.evaluateForest(X_test,y_test) 
print("Accuracy for Forest treeeeeee:", accuracyForest)
'''