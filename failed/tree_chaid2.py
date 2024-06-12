import numpy as np
import pandas as pd

class TreeNode:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

class CHAID:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def calculate_chi_squared(self, left_counts, right_counts):
        # Calculate chi-squared value for splitting
        total_counts = left_counts + right_counts
        expected_left = left_counts.sum() * total_counts / total_counts.sum()
        expected_right = right_counts.sum() * total_counts / total_counts.sum()
        chi_squared = np.sum((left_counts - expected_left)**2 / expected_left + (right_counts - expected_right)**2 / expected_right)
        return chi_squared

    def find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_chi_squared = np.inf
        n_features = X.shape[1]

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            for threshold in unique_values:
                left_indices = X[:, feature_idx] <= threshold
                left_counts = np.bincount(y[left_indices], minlength=2)
                right_counts = np.bincount(y[~left_indices], minlength=2)
                chi_squared = self.calculate_chi_squared(left_counts, right_counts)
                if chi_squared < best_chi_squared:
                    best_chi_squared = chi_squared
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth):
        if depth >= self.max_depth or np.unique(y).size == 1:
            value = np.bincount(y).argmax()
            return TreeNode(value=value)

        best_feature, best_threshold = self.find_best_split(X, y)
        if best_feature is None:
            value = np.bincount(y).argmax()
            return TreeNode(value=value)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_subtree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)

    def predict_single(self, node, sample):
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self.predict_single(node.left, sample)
        else:
            return self.predict_single(node.right, sample)

    def predict(self, X):
        predictions = []
        for sample in X:
            prediction = self.predict_single(self.tree, sample)
            predictions.append(prediction)
        return np.array(predictions)
'''
# Приклад використання
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_train = np.array([0, 0, 1, 1, 1])

model = CHAID(max_depth=3)
model.fit(X_train, y_train)

X_test = np.array([[1, 3], [4, 5]])
predictions = model.predict(X_test)
print(predictions)
'''

# Задані дані
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75],
    'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80],
    'Windy': [False, True, False, False, False, True, True, False, False, False],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Задані класи та мітки
label = 'PlayTennis'
class_list = ['Yes', 'No']

# Підготовка даних
X = data.drop(label, axis=1).values
y = data[label].apply(lambda x: class_list.index(x)).values

# Ваш код для побудови моделі та передбачення результатів
model = CHAID(max_depth=3)
model.fit(X, y)

# Приклад передбачення
new_data = np.array([[2, 85, True], [1, 75, False]])  # Приклад нових даних для передбачення
predictions = model.predict(new_data)
predicted_classes = [class_list[pred] for pred in predictions]
print(predicted_classes)