import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # If only one class in this node or max depth reached, return leaf
        if n_labels == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return {'class': np.argmax(np.bincount(y)), 'n_samples': n_samples}

        # Select random subset of features
        feature_indices = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)

        # Select the best feature and threshold based on information gain
        best_gain = -1
        best_feature, best_threshold = None, None
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        # Split the data based on the best feature and threshold
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': best_feature, 'threshold': best_threshold, 'left': left_tree, 'right': right_tree}

    def _information_gain(self, X, y, feature, threshold):
        parent_entropy = self._entropy(y)
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        n_left, n_right = len(y[left_indices]), len(y[right_indices])
        if n_left == 0 or n_right == 0:
            return 0
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])
        child_entropy = (n_left / len(y)) * left_entropy + (n_right / len(y)) * right_entropy
        return parent_entropy - child_entropy

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def predict(self, X):
        return [self._predict_tree(x, self.tree) for x in X]

    def _predict_tree(self, x, tree):
        if 'class' in tree:
            return tree['class']
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        y_numeric = np.where(y == 'Yes', 1, 0)  # Конвертуємо 'Yes' в 1, 'No' в 0
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y_numeric[indices])
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return self._majority_vote(predictions)

    def _majority_vote(self, predictions):
        most_common = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
        return most_common
    
    # with label
    def evaluate0(self, test_data, label):
        correct_predict = 0
        wrong_predict = 0
        for index, row in test_data.iterrows():
            result = self.predict(np.array([row[:-1]]))  # Передбачення для поточного рядка без мітки класу
            if result[0] == row[label]:  # Порівняння передбачення з міткою класу
                correct_predict += 1
            else:
                wrong_predict += 1
        accuracy = correct_predict / (correct_predict + wrong_predict)
        return accuracy
    
    #without label
    def evaluate(self, test_data):
        correct_predict = 0
        wrong_predict = 0
        for index, row in test_data.iterrows():
            result = self.predict(np.array([row[:-1]]))  # Передбачення для поточного рядка без мітки класу
            if result[0] == row.iloc[-1]:  # Порівняння передбачення з міткою класу останнього стовпця
                correct_predict += 1
            else:
                wrong_predict += 1
        accuracy = correct_predict / (correct_predict + wrong_predict)
        return accuracy
    
    def evaluateForest(self, X_test, y_test):
        predictions = self.predict(X_test)
        print("predictions", predictions)
        bool_arr = np.array(convert_to_bool(predictions))
        print("bool ", bool_arr)
        accuracy = np.mean(bool_arr == y_test)
        print("test ", y_test)
        return accuracy

'''
# Example data
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Example data
test_data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast'],
    'Temperature': [86, 80, 83],
    'Humidity': [85, 91, 78],
    'Windy': [False, True, False],
    'PlayTennis': ['No', 'No', 'Yes']
})
'''

# Ваші дані
X = np.array([
    ['Sunny', 85, 85, False],
    ['Sunny', 80, 90, True],
    ['Overcast', 83, 78, False],
    ['Rain', 70, 96, False],
    ['Rain', 68, 80, False],
    ['Rain', 65, 70, True],
    ['Overcast', 64, 65, True],
    ['Sunny', 72, 95, False],
    ['Sunny', 69, 70, False],
    ['Rain', 75, 80, False]
])

y = np.array(['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes'])

data = pd.DataFrame(X, columns=['Outlook', 'Temperature', 'Humidity', 'Windy'])
data['PlayTennis'] = y
X = data[['Outlook', 'Temperature', 'Humidity', 'Windy']].values
y = data['PlayTennis'].values
# Виведення даних
#print(data)


# Побудова та навчання випадкового лісу
rf = RandomForest(n_estimators=10, max_depth=2)
rf.fit(X, y)

X_test = np.array([
    ['Sunny', 70, 90, False],
    ['Rain', 80, 75, True]
])

def convert_to_bool(arr):
    #return ['yes' if x == 1 else 'no' for x in arr]
    return ['Yes' if x == 1 else 'No' for x in arr]
    #return ['True' if x == 1 else 'False' for x in arr]

y_test = np.array(['False', 'False'])

predictions = rf.predict(X_test)
print("Predictions:", predictions)



#print("Accuracy:", rf.evaluate(new_data))

print("Accuracy:", rf.evaluateForest(X_test,y_test))
