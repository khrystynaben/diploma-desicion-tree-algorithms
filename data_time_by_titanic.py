import numpy as np
import pandas as pd
import time
from newww_c45 import new_c45,new_evaluateС45
from _1_tree_id3 import id3, predictID3, evaluateID3
from _4_new_cart import new_cart, accuracy_new_cart, predict
from _5_tree_chaid4 import chaid_tree, evaluateChaid
from _6_new_forest import accuracy_new_forest, random_forest_learning, random_forest_predict
import sys

def preprocess_data(df):
    # Замінюємо некоректні значення в числових колонках на NaN і потім на середні значення
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Заповнюємо NaN середніми значеннями
    df.fillna(df.mean(), inplace=True)
    
    return df



#--------------------Titanic-(800 100)----------------------
train_data_titanic = pd.read_csv("D:/university/diploma/data/titanic/re_train_withoutname_id.csv") 
print(train_data_titanic)
test_data_titanic = pd.read_csv("D:/university/diploma/data/titanic/re_test_withoutname_id.csv") 


# ----------------1--ID3-----------------------
start_time1 = time.time()  # Початок вимірювання часу
treeID3 = id3(train_data_titanic, 'Survived')
end_time1 = time.time()  # Кінець вимірювання часу
execution_time1 = end_time1 - start_time1
print(f"Час виконання алгоритму ID3: {execution_time1} секунд")
print(treeID3)
accuracyID3 = evaluateID3(treeID3, test_data_titanic, 'Survived') 
print("Accuracy for ID3 treeeeeee:", accuracyID3)



# Попередня обробка даних
df = preprocess_data(train_data_titanic)
# ----------------2--C45-----------------------

X_emp = df.drop(columns=['Survived']).values
y_emp = df['Survived'].values
X_emp_test = test_data_titanic.drop(columns=['Survived']).values
start_time2 = time.time()  # Початок вимірювання часу
#treeC45 = c45(train_data_tennis, 'Play Tennis')
treeC45 = new_c45(X_emp, y_emp, min_samples_split=2, max_depth=None)
end_time2 = time.time()  # Кінець вимірювання часу
execution_time2 = end_time2 - start_time2
print(f"Час виконання алгоритму C45: {execution_time2} секунд")
print(treeC45)
accuracyC45 = new_evaluateС45(treeC45, test_data_titanic, 'Survived') 
print("Accuracy for C45 treeeeeee:", accuracyC45)



# ----------------4--CART-----------------------

start_time3 = time.time()  # Початок вимірювання часу
#check if # does not matter 100 1000
treeCart = new_cart(df.values.tolist(), 100, 1)
print(treeCart)
end_time3 = time.time()  # Кінець вимірювання часу
execution_time3 = end_time3 - start_time3
print(f"Час виконання алгоритму CART: {execution_time3} секунд")
predictions = []
for index, row in test_data_titanic.iterrows():
    prediction = predict(treeCart, row)
    predictions.append(prediction)

# Приклад використання
actual_values = test_data_titanic['Survived'].values.tolist()  # Справжні значення міток
accuracy = accuracy_new_cart(actual_values, predictions)  # predictions - передбачені значення
print('Accuracy for Cart:', accuracy)



# -----------------5-CHAID-----------------------
#1.0
label = 'Survived'
class_list = ['0', '1']
start_time5 = time.time()  # Початок вимірювання часу
treeChaid = chaid_tree(train_data_titanic,  label, class_list)
end_time5 = time.time()  # Кінець вимірювання часу
print(treeChaid)
execution_time5 = end_time5 - start_time5
print(f"Час виконання алгоритму Chaid: {execution_time5} секунд")
accuracyChaid = evaluateChaid(treeChaid, test_data_titanic, 'Survived') 
print("Accuracy for Chaid treeeeeee:", accuracyChaid)


# -----------------6-Forest-----------------------
# change data
predictions = []
data = train_data_titanic.values.tolist()
test_data = test_data_titanic.values.tolist()
start_time4 = time.time()  # Початок вимірювання часу
trees = random_forest_learning(data, 10)
print(trees)
end_time4 = time.time()  # Кінець вимірювання часу
execution_time4 = end_time4 - start_time4
print(f"Час виконання алгоритму Forest: {execution_time4} секунд")
for example in test_data:
    prediction = random_forest_predict(trees, example)
    predictions.append(prediction)
    #print(f"Example: {example}, Prediction: {prediction}")

#print('Prediction:', predictions)

actual_values = test_data_titanic['Survived'].values.tolist()  # Справжні значення міток
accuracy = accuracy_new_forest(actual_values, predictions)  # predictions - передбачені значення
print('Accuracy for Forest :', accuracy)

'''
#!!!!!predictions все 0
treeForest = RandomForest(n_estimators=100, max_depth=10)
X_train = train_data_emp[['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age','Gender','EverBenched','ExperienceInCurrentDomain']].values
y_train = train_data_emp['LeaveOrNot'].values
treeForest.fit(X_train,y_train)

X_test = test_data_emp[['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age','Gender','EverBenched','ExperienceInCurrentDomain']].values
y_test = test_data_emp['LeaveOrNot'].values
accuracyForest = treeForest.evaluateForest(X_test,y_test) 
print("Accuracy for Forest treeeeeee:", accuracyForest)
'''
