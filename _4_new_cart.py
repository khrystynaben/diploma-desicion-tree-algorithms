import pandas as pd

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # індекс ознаки для розбиття
        self.threshold = threshold  # поріг розбиття
        self.left = left  # ліве піддерево
        self.right = right  # праве піддерево
        self.value = value  # значення листка (клас або прогноз)

def calculate_gini_index(groups, classes):
    # обчислює Gini Index для розбиття
    # використовується для визначення кращого способу розбиття
    n_instances = float(sum(len(group) for group in groups))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

def split_dataset(index, value, dataset):
    left, right = [], []
    for row in dataset:
        #print(row[index] , ' ', value)
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = float('inf'), float('inf'), float('inf'), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_dataset(index, row[index], dataset)
            gini = calculate_gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right)
        split(node['right'], max_depth, min_size, depth + 1)

def new_cart(train, max_depth, min_size):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    #print("_first_ ", row[node['index']] , "_second_ ",node['value'])
    if row[node['index']] < node['value']:
        
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def accuracy_new_cart(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) 

'''
train_data_marks = pd.read_csv("D:/train/tree_data_marks.csv") 
#print(train_data_marks)
test_data_marks = pd.read_csv("D:/test/tree_data_marks.csv") 
#train_data_tennis = pd.read_csv("D:/train/PlayTennis.csv") 
#test_data_tennis = pd.read_csv("D:/test/PlayTennis.csv") 

#train_data_emp = pd.read_csv('emp_data/emp_train_4000.csv')  
#test_data_emp = pd.read_csv('emp_data/emp_test_600.csv') 
tree = new_cart(train_data_marks.values.tolist(), 100, 1)
#print(tree)

predictions = []
for index, row in test_data_marks.iterrows():
    prediction = predict(tree, row)
    predictions.append(prediction)

#prediction = predict(tree, test_data_marks.values.tolist())
print('Prediction:', predictions)


actual_values = test_data_marks['mark'].values.tolist()  # Справжні значення міток
accuracy = accuracy_new_cart(actual_values, predictions)  # predictions - передбачені значення
print('Accuracy:', accuracy)
'''

