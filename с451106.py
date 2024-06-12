import pandas as pd
import numpy as np

class Node:
    def __init__(self, attribute=None, threshold=None, label=None, is_leaf=False):
        self.attribute = attribute
        self.threshold = threshold
        self.label = label
        self.is_leaf = is_leaf
        self.children = []

def entropy(y):
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

def prune_tree(node, X, y):
    if not node.children:
        return node
    
    X_left, X_right, y_left, y_right = split_data(X, y, node.attribute, node.threshold)
    
    node.children[0] = prune_tree(node.children[0], X_left, y_left)
    node.children[1] = prune_tree(node.children[1], X_right, y_right)
    
    # Calculate current node's error without splitting
    error_no_split = np.sum(y != node.label) / len(y)
    
    # Calculate error after splitting
    error_split = (
        len(y_left) * np.sum(y_left != np.bincount(y_left).argmax()) +
        len(y_right) * np.sum(y_right != np.bincount(y_right).argmax())
    ) / len(y)
    
    # Prune if splitting doesn't improve accuracy
    if error_split >= error_no_split:
        node.children = []
        node.is_leaf = True
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

def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    return correct_predictions / total_samples


 
    #--------------------Employee-(4 000 600)----------------------
train_data_emp = pd.read_csv('data/emp_data/emp_train_4000.csv')  
test_data_emp = pd.read_csv('data/emp_data/emp_test_600.csv') 

X_emp = train_data_emp.drop(columns=['LeaveOrNot']).values
y_emp = train_data_emp['LeaveOrNot'].values
X_emp_test = train_data_emp.drop(columns=['LeaveOrNot']).values

    

# Usage example
# Assuming X_train, y_train are your training data
tree = new_c45(X_emp, y_emp)
pruned_tree = prune_tree(tree, X_emp, y_emp)

# Example usage
# Assuming X_test, y_test are your test data
y_pred = predict(pruned_tree, X_emp_test)
acc = accuracy(y_emp, y_pred)
print(y_emp)
print(y_pred)
print(f"Accuracy: {acc}")

