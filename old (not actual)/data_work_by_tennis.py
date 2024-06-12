import numpy as np
import pandas as pd
from _1_tree_id3 import id3, predictID3, evaluateID3
from _2_tree_c45 import c45, evaluateС45
from _3_tree_c50 import c50, evaluateC50
from _4_tree_cart import cart, evaluateCart
from _5_tree_chaid4 import chaid_tree, evaluateChaid
from _6_tree_forest import RandomForest
import sys

#sys.setrecursionlimit(30000)

# Зчитування даних з файлів
#--------------------Employee-(4 000 600)----------------------
train_data_emp = pd.read_csv('emp_data/emp_train_4000.csv')  
test_data_emp = pd.read_csv('emp_data/emp_test_600.csv') 

#--------------------Play Tennis-(15 5)----------------------
train_data_tennis = pd.read_csv("D:/train/PlayTennis.csv") 
test_data_tennis = pd.read_csv("D:/test/PlayTennis.csv") 



# ----------------1--ID3-----------------------
# 0.46 для не повторюваних даних 200 50 make_tree
# 0.46 для повторюваних даних 200 50 make_tree2
# 0.57 для повторюваних даних 4000 600 make_tree2
#tree_id3_emp = id3(train_data_emp, 'LeaveOrNot')
treeID3 = id3(train_data_tennis, 'Play Tennis')
print(treeID3)


#accuracy_emp = evaluateID3(tree, test_data_emp, 'LeaveOrNot')
accuracyID3 = evaluateID3(treeID3, test_data_tennis, 'Play Tennis') 
print("Accuracy for ID3 treeeeeee:", accuracyID3)


# ----------------2--C45-----------------------
#tree = c45(train_data_emp, 'LeaveOrNot')
treeC45 = c45(train_data_tennis, 'Play Tennis')
print(treeC45)

#accuracy_emp = evaluateC45(tree, test_data_emp, 'LeaveOrNot')
accuracyC45 = evaluateС45(treeC45, test_data_tennis, 'Play Tennis') 
print("Accuracy for C45 treeeeeee:", accuracyC45)

'''
# ----------------3--C50-----------------------
#tree = c45(train_data_emp, 'LeaveOrNot')
treeC50 = c50(train_data_tennis, 'Play Tennis')
print(treeC50)

#accuracy_emp = evaluateC50(tree, test_data_emp, 'LeaveOrNot')
accuracyC50 = evaluateC50(treeC50, test_data_tennis) 
print("Accuracy for C50 treeeeeee:", accuracyC50)
'''


# ----------------4--CART-----------------------
#tree = cart(train_data_emp, 'LeaveOrNot')
label = 'Play Tennis'
class_list = ['Yes', 'No']
treeCart = cart(train_data_tennis, label, class_list)
#print(treeCart)

#accuracy_emp = evaluateID3(tree, test_data_emp, 'LeaveOrNot')
accuracyCart = evaluateCart(treeCart, test_data_tennis, 'Play Tennis') 
print("Accuracy for Cart treeeeeee:", accuracyCart)


# -----------------5-CHAID-----------------------
#tree = chaid_tree(train_data_emp, 'LeaveOrNot')
treeChaid = chaid_tree(train_data_tennis,  label, class_list)
#print(treeChaid)
#accuracy_emp = evaluateID3(tree, test_data_emp, 'LeaveOrNot')
accuracyChaid = evaluateChaid(treeChaid, test_data_tennis, 'Play Tennis') 
print("Accuracy for Chaid treeeeeee:", accuracyChaid)


# -----------------6-Forest-----------------------
#tree = c45(train_data_emp, 'LeaveOrNot')
treeForest = RandomForest(n_estimators=10, max_depth=2)
X_train = train_data_tennis[['Outlook', 'Temperature', 'Humidity', 'Wind']].values
y_train = train_data_tennis['Play Tennis'].values
treeForest.fit(X_train,y_train)
#print(treeForest)

#accuracy_emp = evaluateID3(tree, test_data_emp, 'LeaveOrNot')
X_test = test_data_tennis[['Outlook', 'Temperature', 'Humidity', 'Wind']].values
y_test = test_data_tennis['Play Tennis'].values
accuracyForest = treeForest.evaluateForest(X_test,y_test) 
print("Accuracy for Forest treeeeeee:", accuracyForest)
