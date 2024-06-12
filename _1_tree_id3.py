import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0] #кількість рядків
    total_entr = 0
    
    for c in class_list: 
        total_class_count = train_data[train_data[label] == c].shape[0] #кількість класів
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row) #ентропія класу
        total_entr += total_class_entr 
    
    return total_entr


def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
    
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]  
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count #p
            entropy_class = - probability_class * np.log2(probability_class)  #entropy
        entropy += entropy_class
    return entropy


def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique() #унікалбні знач
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value] #фільтр на конкретне значння
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list) 
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy #p * E
        
    return calc_total_entropy(train_data, label, class_list) - feature_info #цінність інформації


def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) #список елементів
                                            
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain: #пошук з найб цінністю
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature


def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False) #кільість елем
    tree = {} 
    
    for feature_value, count in feature_value_count_dict.items():
    #for feature_value, count in feature_value_count_dict.iteritems():
        feature_value_data = train_data[train_data[feature_name] == feature_value] # only feature_name = feature_value
        
        assigned_to_node = False #мітка для чистого класу
        for c in class_list: 
            class_count = feature_value_data[feature_value_data[label] == c].shape[0] 

            if class_count == count: #кількість (feature_value = count) 
                tree[feature_value] = c #додаємо вузли
                train_data = train_data[train_data[feature_name] != feature_value] #уникнення повторення
                assigned_to_node = True
        if not assigned_to_node: #не чистий клас
            tree[feature_value] = "?" #не чистий клас, тому невідомо
                                      
            
    return tree, train_data


#for file without copy
def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0: #якщо має вміст
        max_info_feature = find_most_informative_feature(train_data, label, class_list) 
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list) #додавання вузла,оновлення дерева
        next_root = None
        
        if prev_feature_value != None: #додавання до ?
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else: #додавання початкового
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        
        for node, branch in list(next_root.items()): #iterating the tree node
            if branch == "?": #if it is expandable
                feature_value_data = train_data[train_data[max_info_feature] == node] #оновлені дані
                print(feature_value_data)
                make_tree(next_root, node, feature_value_data, label, class_list) #рекурсивність


#для даних із повторюваними значеннями
def make_tree2(root, prev_feature_value, train_data, label, class_list, processed_values=None):
    if processed_values is None:
        processed_values = set()  # Створюємо множину для збереження оброблених значень
    
    if train_data.shape[0] != 0: 
        max_info_feature = find_most_informative_feature(train_data, label, class_list) 
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list)
        
        if tree:  
            next_root = None
            
            if prev_feature_value is not None:
                root[prev_feature_value] = dict()
                root[prev_feature_value][max_info_feature] = tree
                next_root = root[prev_feature_value][max_info_feature]
            else:
                root[max_info_feature] = tree
                next_root = root[max_info_feature]
            
            if next_root is not None and isinstance(next_root, dict): 
                for node, branch in list(next_root.items()):
                    if branch == "?":
                        feature_value_data = train_data[train_data[max_info_feature] == node]
                        feature_value_hash = hash(feature_value_data.to_string())
                        if feature_value_hash not in processed_values:  # Перевірка на наявність обробленого значення
                            processed_values.add(feature_value_hash)  # Додаємо хеш унікального значення
                            make_tree2(next_root, node, feature_value_data, label, class_list, processed_values)



def id3(train_data_m, label):
    train_data = train_data_m.copy() 
    tree = {} 
    class_list = train_data[label].unique() 
    make_tree2(tree, None, train_data, label, class_list) 
    return tree


def predictID3(tree, instance):
    if not isinstance(tree, dict): #якщо не листок
        return tree 
    else:
        root_node = next(iter(tree)) #починаємо з корен елем
        feature_value = instance[root_node] #значення
        if feature_value in tree[root_node]: #чи знач існує
            return predictID3(tree[root_node][feature_value], instance) #наступна характ
        else:
            return None


def evaluateID3(tree, test_data_m, label):
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows(): 
        result = predictID3(tree, test_data_m.iloc[index])
        if result == test_data_m[label].iloc[index]: 
            correct_preditct += 1 
        else:
            wrong_preditct += 1 
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) 
    return accuracy

'''..............not working.........................
def plot_decision_tree(tree, ax=None, position=None, level=0):
    if ax is None:
        ax = plt.gca()

    if position is None:
        position = np.array([0.5, 1])

    if isinstance(tree, str):
        ax.text(position[0], position[1], tree, ha='center', va='center', 
                bbox=dict(facecolor='white', edgecolor='blue'))
        return

    node = list(tree.keys())[0]
    children = tree[node]

    ax.text(position[0], position[1], node, ha='center', va='center', 
            bbox=dict(facecolor='white', edgecolor='green'))

    level += 1
    height = len(children)
    #print("len^ ", len(children))
    width = 1 / height

    for i, child in enumerate(children):
        x = position[0] - 0.5 + i * width + 0.5 * width
        y = position[1] - 0.5

        ax.plot([position[0], x], [position[1], y], 'k-')
        ax.text(x, y, child, ha='center', va='center', 
                bbox=dict(facecolor='white', edgecolor='pink'))

        plot_decision_tree(children[child], ax=ax, position=np.array([x, y]), level=level)

    if level == 1:
        ax.axis('off')
......................................................'''



#-----------------years--------------------------
#1train_data_m = pd.read_csv("D:/train/tree_data_years.csv") 
#2print(train_data_m)
#3tree = id3(train_data_m, 'result')


#-----------------mark--------------------------
#train_data_m = pd.read_csv("D:/train/tree_data_marks.csv") 
#print(train_data_m)
#tree = id3(train_data_m, 'mark')

'''
#--------------------Play Tennis-----------------------
train_data_m = pd.read_csv("D:/train/PlayTennis.csv") 
tree = id3(train_data_m, 'Play Tennis')
print(tree)
test_data_m = pd.read_csv("D:/test/PlayTennis.csv") 
# Оцінка точності моделі на тестових даних
accuracy = evaluateID3(tree, test_data_m, 'Play Tennis') 
print("Accuracy for tree:", accuracy)
'''

'''..............not working.........................
# Створення графіку
fig, ax = plt.subplots(figsize=(6, 6))
plot_decision_tree(tree, ax=ax)

plt.show()
#................................................'''


'''
......................WORKING.........................
import pydot

def plot_decision_tree2(tree, parent_node=None, parent_edge_label=None):
    if isinstance(tree, str):
        leaf_node = pydot.Node(tree, shape='box')
        graph.add_node(leaf_node)
        if parent_node is not None:
            edge = pydot.Edge(parent_node, leaf_node, label=parent_edge_label)
            graph.add_edge(edge)
    else:
        for node_label, subtree in tree.items():
            node = pydot.Node(node_label, shape='box')
            graph.add_node(node)
            if parent_node is not None:
                edge = pydot.Edge(parent_node, node, label=parent_edge_label)
                graph.add_edge(edge)
            plot_decision_tree2(subtree, parent_node=node, parent_edge_label=node_label)
##.----------------- створення дерева ------------------
import os
os.environ["PATH"] += os.pathsep + 'D:/university/3_course/course_work/Graphviz/bin/'

# Створення графу
graph = pydot.Dot(graph_type='graph')
plot_decision_tree2(tree)


## Збереження графу у файл
graph.write_png('decision_tree.png')
#-------------------------------------------------------------
'''


#------------------------------years-------------------------------------
#test_data_m = pd.read_csv("D:/test/tree_data_years.csv") 
#accuracy = evaluate(tree, test_data_m, 'result') 
#print("Accuracy for tree:", accuracy)


#------------------------------mark-------------------------------------
#test_data_m = pd.read_csv("D:/test/tree_data_marks.csv") 
#accuracy = evaluate(tree, test_data_m, 'mark') 
#print("Accuracy for tree:", accuracy)


#----------------------------Play Tennis ---------------------------------------
#test_data_m = pd.read_csv("D:/test/PlayTennis.csv") 
#accuracy = evaluate(tree, test_data_m, 'Play Tennis') 

#прохідний бал, чи збільшилась кількісь бм, ккість заяв, конкурс на бюджет, середній пріорітет допущений, зовн обставини?, зміна умов?