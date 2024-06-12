import numpy as np
import pandas as pd
import time

class CHAIDNode:
    def __init__(self, data, depth=0, max_depth=None):
        self.data = data
        self.depth = depth
        self.max_depth = max_depth
        self.children = []
        self.split_col = None
        self.split_val = None
        self.is_leaf = True
        self.label = self.get_label()

    def get_label(self):
        labels, counts = np.unique(self.data.iloc[:, -1], return_counts=True)
        return labels[np.argmax(counts)]

    def chi_square_test(self, col):
        # Створення таблиці спостережень
        contingency_table = pd.crosstab(self.data[col], self.data.iloc[:, -1])
        observed = contingency_table.values

        # Обчислення очікуваних частот
        row_totals = observed.sum(axis=1)
        col_totals = observed.sum(axis=0)
        total = observed.sum()
        expected = np.outer(row_totals, col_totals) / total

        # Обчислення χ²
        chi2 = ((observed - expected) ** 2 / expected).sum()
        return chi2, observed.size - 1

    def best_split(self):
        best_p = 0
        best_col = None
        for col in self.data.columns[:-1]:
            chi2, dof = self.chi_square_test(col)
            if chi2 > best_p:
                best_p = chi2
                best_col = col
        return best_col, best_p

    def split(self):
        if self.max_depth is not None and self.depth >= self.max_depth:
            return
        self.split_col, chi2_value = self.best_split()
        if chi2_value > self.critical_value():
            self.is_leaf = False
            for val in np.unique(self.data[self.split_col]):
                child_data = self.data[self.data[self.split_col] == val]
                child_node = CHAIDNode(child_data, self.depth + 1, self.max_depth)
                child_node.split()
                self.children.append((val, child_node))

    def critical_value(self, alpha=0.05):
        # Таблиця критичних значень χ² для різних рівнів значущості
        chi2_critical_values = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070}
        return chi2_critical_values.get(self.children_dof(), 3.841)

    def children_dof(self):
        if self.split_col is None:
            return 0
        unique_vals = np.unique(self.data[self.split_col])
        return len(unique_vals) - 1

    def predict(self, x):
        if self.is_leaf:
            return self.label
        for val, child in self.children:
            if x[self.split_col] == val:
                return child.predict(x)
        return self.label

class CHAID:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, data):
        self.root = CHAIDNode(data, max_depth=self.max_depth)
        self.root.split()

    def predict_old(self, X):
        return X.apply(lambda x: self.root.predict(x), axis=1)
    
    def predict(self, X):
        predictions = []
        for index, row in X.iterrows():
            prediction = self.root.predict(row)
            predictions.append(prediction)
        return predictions
    
def accuracy_chaid(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual))     

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
test_data4= pd.read_csv('data/emp_data/emp_test_600.csv') 
label4 = 'LeaveOrNot'

#--------------------Satisfaction---------------------
train_data = pd.read_csv('data/satisfaction/train20000.csv')  
test_data = pd.read_csv('data/satisfaction/test4000.csv') 
label = 'satisfaction'

'''
# Приклад використання:
data = pd.DataFrame({
    'Age': ['<=30', '<=30', '31-40', '>40', '>40'],
    'Income': ['High', 'Medium', 'High', 'Low', 'High'],
    'Student': ['No', 'Yes', 'Yes', 'No', 'Yes'],
    'Buy': ['No', 'Yes', 'Yes', 'No', 'Yes']
})
test_data = pd.DataFrame({
    'Age': ['<=30', '31-40', '>40'],
    'Income': ['Medium', 'High', 'Low'],
    'Student': ['Yes', 'No', 'Yes']
})

chaid = CHAID(max_depth=None)
chaid.fit(data)

predictions = chaid.predict(test_data)
print(predictions)
'''


'''
start_time1 = time.time()  # Початок вимірювання часу


chaid2 = CHAID(max_depth=None)
chaid2.fit(train_data)

end_time1 = time.time()  # Кінець вимірювання часу
execution_time1 = end_time1 - start_time1
print(f"Час виконання алгоритму Chaid: {execution_time1} секунд")

predictions2 = chaid2.predict(test_data)
print(predictions2)
actual = test_data[label].values.tolist()
print(actual)

print(type(predictions2))
print(accuracy_chaid(actual, predictions2))
'''