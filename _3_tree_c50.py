import numpy as np
import pandas as pd

# Sample dataset (replace this with your actual dataset)
data0 = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': ['A', 'B', 'A', 'B', 'A'],
    'Target': ['Yes', 'No', 'Yes', 'No', 'Yes']
}
target = 'Play Tennis'

# Function to calculate entropy
def entropy(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Function to split data based on a feature and its value
def split_data(data, feature, value):
    subset1 = {key: [] for key in data.keys()}
    subset2 = {key: [] for key in data.keys()}
    for i in range(len(data[feature])):
        if data[feature][i] == value:
            for key in data.keys():
                subset1[key].append(data[key][i])
        else:
            for key in data.keys():
                subset2[key].append(data[key][i])
    return subset1, subset2

# Function to find the best split based on information gain
def find_best_split(data):
    features = list(data.keys())[:-1]  # Exclude the target variable
    best_gain = 0
    best_feature = None
    best_value = None
    base_entropy = entropy(data[target])
    for feature in features:
        unique_values = np.unique(data[feature])
        for value in unique_values:
            subset1, subset2 = split_data(data, feature, value)
            subset1_entropy = entropy(subset1[target])
            subset2_entropy = entropy(subset2[target])
            weighted_entropy = (len(subset1[target]) / len(data[target])) * subset1_entropy + \
                               (len(subset2[target]) / len(data[target])) * subset2_entropy
            information_gain = base_entropy - weighted_entropy
            if information_gain > best_gain:
                best_gain = information_gain
                best_feature = feature
                best_value = value
    return best_feature, best_value

# Recursive function to build the decision tree
def c50(data):
    if len(np.unique(data[target])) == 1:
        return np.unique(data[target])[0]
    if len(data.keys()) == 1:
        return None
    best_feature, best_value = find_best_split(data)
    subset1, subset2 = split_data(data, best_feature, best_value)
    decision_tree = {best_feature: {}}
    decision_tree[best_feature][best_value] = c50(subset1)
    decision_tree[best_feature]['Not ' + str(best_value)] = c50(subset2)
    return decision_tree

# Example usage
#decision_tree = C50(data)
#print(decision_tree)

# Function to predict the target variable using the decision tree
def predict(decision_tree, instance):
    feature = list(decision_tree.keys())[0]
    value = instance[feature]
    if value in decision_tree[feature]:
        subtree = decision_tree[feature][value]
    else:
        subtree = decision_tree[feature]['Not ' + str(value)]
    if isinstance(subtree, dict):
        return predict(subtree, instance)
    else:
        return subtree

# Function to evaluate the decision tree using accuracy
def evaluateC50(decision_tree, test_data):
    predictions = [predict(decision_tree, instance) for _, instance in test_data.iterrows()]
    actual = test_data[target].tolist()
    correct = sum(1 for pred, actual in zip(predictions, actual) if pred == actual)
    accuracy = correct / len(test_data)
    return accuracy
'''
# Example usage
decision_tree = c50(data0)
print("Decision Tree:", decision_tree)

# Test data for evaluation
test_data = pd.DataFrame({
    'Feature1': [3, 4, 1, 5],
    'Feature2': ['A', 'B', 'A', 'A'],
    'Target': ['Yes', 'No', 'Yes', 'Yes']
})

# Evaluate the decision tree using test data
accuracy = evaluateC50(decision_tree, test_data)
print("Accuracy:", accuracy)
'''
'''
# Function to predict using the decision tree
def predictC50(tree, instance):
    if not isinstance(tree, dict):  # Leaf node
        return tree
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predictC50(tree[root_node][feature_value], instance)
        else:
            return None

# Function to evaluate the decision tree
def evaluateC50(tree, test_data, label):
    correct_predict = 0
    wrong_predict = 0
    for index, row in test_data.iterrows():
        result = predictC50(tree, test_data.iloc[index])
        if result == test_data[label].iloc[index]:
            correct_predict += 1
        else:
            wrong_predict += 1
    accuracy = correct_predict / (correct_predict + wrong_predict)
    return accuracy

# Example usage:
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})


target = 'PlayTennis'
# Example usage
decision_tree = c50(data)
print(decision_tree)

# Example usage of predict and evaluate
test_data = pd.DataFrame({
    'Outlook': ['Sunny', 'Overcast', 'Rain', 'Sunny'],
    'Temperature': [75, 70, 85, 90],
    'Humidity': [70, 90, 80, 85],
    'Windy': [False, True, True, False],
    'PlayTennis': ['Yes', 'Yes', 'No', 'No']  # Actual outcomes for evaluation
})

accuracy = evaluateC50(decision_tree, test_data, 'PlayTennis')
print("Accuracy:", accuracy)
'''