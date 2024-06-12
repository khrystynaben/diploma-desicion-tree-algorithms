import pandas as pd
import numpy as np
import time

class Node:
    def __init__(self, attribute=None, threshold=None, label=None, is_leaf=False):
        self.attribute = attribute
        self.threshold = threshold
        self.label = label
        self.is_leaf = is_leaf
        self.children = []

'''
def convert_to_bool(arr):
    return [1 if x == 'yes' else 0 for x in arr]
'''

def entropy(y):
    #y = convert_to_bool(y)
    #print(y)
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def information_gain_ratio(y, y_subsets):
    total_entropy = entropy(y)
    subset_entropy = 0
    split_info = 0
    for subset in y_subsets:
        proportion = len(subset) / len(y)
        subset_entropy += proportion * entropy(subset)
        split_info -= proportion * np.log2(proportion)
    info_gain = total_entropy - subset_entropy
    return info_gain / split_info if split_info != 0 else 0

def split_data(X, y, attribute, threshold):
    left_mask = X[:, attribute] <= threshold
    right_mask = ~left_mask
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

def best_split(X, y):
    best_gain_ratio = -1
    best_attribute = None
    best_threshold = None
    for attribute in range(X.shape[1]):
        thresholds = np.unique(X[:, attribute])
        for threshold in thresholds:
            _, _, y_left, y_right = split_data(X, y, attribute, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            gain_ratio = information_gain_ratio(y, [y_left, y_right])
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_attribute = attribute
                best_threshold = threshold
    return best_attribute, best_threshold

def new_c45(X, y, min_samples_split=2, depth=0, max_depth=None):
    if len(np.unique(y)) == 1 or len(y) < min_samples_split or (max_depth is not None and depth >= max_depth):
        return Node(label=np.bincount(y).argmax(), is_leaf=True)
    
    attribute, threshold = best_split(X, y)
    if attribute is None:
        return Node(label=np.bincount(y).argmax(), is_leaf=True)
    
    X_left, X_right, y_left, y_right = split_data(X, y, attribute, threshold)
    if len(y_left) == 0 or len(y_right) == 0:
        return Node(label=np.bincount(y).argmax(), is_leaf=True)

    left_child = new_c45(X_left, y_left, min_samples_split, depth + 1, max_depth)
    right_child = new_c45(X_right, y_right, min_samples_split, depth + 1, max_depth)
    node = Node(attribute=attribute, threshold=threshold)
    node.children = [left_child, right_child]
    return node

def predict_single(node, x):
    if node.is_leaf:
        return node.label
    if x[node.attribute] <= node.threshold:
        return predict_single(node.children[0], x)
    else:
        return predict_single(node.children[1], x)

def predict(tree, X):
    return [predict_single(tree, x) for x in X]

def new_evaluateС45(tree, test_data_m, label):
    my_list = []
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows(): 
        result = predict_single(tree, test_data_m.iloc[index])
        my_list.append(result)
        if result == test_data_m[label].iloc[index]: 
            correct_preditct += 1 
        else:
            wrong_preditct += 1 
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) 
    #print("check prediction: ", my_list)
    return accuracy

# Example usage
if __name__ == "__main__":
    '''
    data = {
        'math': [80, 85, 78, 65, 72, 90],
        'physics': [70, 88, 82, 75, 68, 85],
        'chemistry': [90, 80, 85, 72, 78, 92],
        'additional': [8, 9, 7, 6, 8, 10],
        'attendance': [95, 98, 88, 80, 85, 100],
        'mark': [1, 1, 1, 0, 0, 1]
    }

    df = pd.DataFrame(data)
    X = df.drop(columns=['mark']).values
    y = df['mark'].values

    tree = new_c45(X, y, min_samples_split=2, max_depth=None)
    predictions = predict(tree, X)
    print("Predictions:", predictions)
    print("Actual:", y.tolist())
    '''
    
    '''
    #--------------------Employee-(4 000 600)----------------------
    train_data_emp = pd.read_csv('emp_data/emp_train_4000.csv')  
    test_data_emp = pd.read_csv('emp_data/emp_test_600.csv') 

    X_emp = train_data_emp.drop(columns=['LeaveOrNot']).values
    y_emp = train_data_emp['LeaveOrNot'].values
    X_emp_test = train_data_emp.drop(columns=['LeaveOrNot']).values

    treeC45 = new_c45(X_emp, y_emp, min_samples_split=2, max_depth=None)
    print(treeC45)
    predictions = predict(treeC45, X_emp_test)
    print("Predictions:", predictions)
    print("Actual:", y_emp.tolist())
    accuracyC45 = new_evaluateС45(treeC45, test_data_emp, 'LeaveOrNot') 
    print("Accuracy for C45 treeeeeee:", accuracyC45)
    '''
    '''
    #--------------------Marks-(115 15)----------------------
    train_data_marks = pd.read_csv("D:/university/diploma/data/marks/new_train.csv") 
    test_data_marks = pd.read_csv("D:/university/diploma/data/marks/new_test.csv") 
    # ----------------2--C45-----------------------
    X_emp = train_data_marks.drop(columns=['mark']).values
    y_emp = train_data_marks['mark'].values
    #y_emp = convert_to_bool(y_emp)
    X_emp_test = test_data_marks.drop(columns=['mark']).values
    start_time2 = time.time()  # Початок вимірювання часу
    #treeC45 = c45(train_data_marks, 'mark')
    treeC45 = new_c45(X_emp, y_emp, min_samples_split=2, max_depth=None)
    end_time2 = time.time()  # Кінець вимірювання часу
    execution_time2 = end_time2 - start_time2
    print(f"Час виконання алгоритму C45: {execution_time2} секунд")
    #print(treeC45)
    accuracyC45 = new_evaluateС45(treeC45, test_data_marks, 'mark') 
    print("Accuracy for C45 treeeeeee:", accuracyC45)
    '''