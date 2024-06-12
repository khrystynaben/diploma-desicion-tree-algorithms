import matplotlib.pyplot as plt
import numpy as np

'''
def x_y(N, alpha_,beta_):
    x = np.linspace(1, N, N)
    y = []
    for i in range(N):
        y.append(alpha_*x[i] + beta_)

    return x,y
'''

def plot_graph(x_label, y_label, title, alpha_, beta_,n,arr_x,arr_y):

    upper_arr_y = []
    lower_arr_y = []

    upper_arr_x = []
    lower_arr_x = []

    x = np.linspace(1, n, n)
    y = []
    for i in range(n):
        y.append(alpha_*x[i] + beta_)

    for i in range(n):
        print(arr_y[i],y[i])
        if(arr_y[i]<=y[i]):
            lower_arr_x.append(arr_x[i])
            lower_arr_y.append(arr_y[i])
        else:
            upper_arr_x.append(arr_x[i])
            upper_arr_y.append(arr_y[i])

    print(lower_arr_y)
    print(upper_arr_y)


    plt.scatter(upper_arr_x, upper_arr_y, Color = "green")
    plt.scatter(lower_arr_x, lower_arr_y, Color = "red")
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.show()




def plot_graph2(x_label, y_label, title, alpha_, beta_, test_data, results_for_test_data,n,arr_x,arr_y):

    upper_arr_y = []
    lower_arr_y = []

    upper_arr_x = []
    lower_arr_x = []

    N = n + len(test_data)

    x = np.linspace(1, N, N)
    y = []
    for i in range(N):
        y.append(alpha_*x[i] + beta_)

    #x = x_y(N, alpha_,beta_)[0]
    #y = x_y(N, alpha_,beta_)[1]
    for i in range(n):
        print(arr_y[i],y[i])
        if(arr_y[i]<=y[i]):
            lower_arr_x.append(arr_x[i])
            lower_arr_y.append(arr_y[i])
        else:
            upper_arr_x.append(arr_x[i])
            upper_arr_y.append(arr_y[i])
    print(lower_arr_y)
    print(upper_arr_y)

    plt.scatter(test_data, results_for_test_data, Color = "purple")
    #plt.scatter(test_data, predict(test_data, alpha_, beta_,n), Color = "pink")
    plt.scatter(upper_arr_x, upper_arr_y, Color = "green")
    plt.scatter(lower_arr_x, lower_arr_y, Color = "red")
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.show()


def regression(arr_x, arr_y , n):
    x_average = sum(arr_x)/n
    y_average = sum(arr_y)/n

    xy = 0
    xx = 0
    yy = 0

    for i in range(n):
        xy += arr_x[i]* arr_y[i]
        xx += arr_x[i]* arr_x[i]
        yy += arr_y[i]* arr_y[i]

    alpha = (xy - n * x_average * y_average)/(xx - n*x_average*x_average)
    beta = y_average - alpha * x_average
    return alpha, beta


def evaluate(test_data,alpha_,beta_,results_for_test_data,n):
    correct_preditct = 0
    wrong_preditct = 0
    result_y = predict(test_data, alpha_,beta_,n) #predict the row
    for i in range(len(test_data)):     
        if abs(result_y[i] - results_for_test_data[i])<3:
            correct_preditct += 1 #increase correct count
        else:
            wrong_preditct += 1 #increase incorrect count
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) #calculating accuracy
    return accuracy


def predict(test_data,alpha_,beta_, n): #допис діапазон
    N = n+len(test_data)
    x = np.linspace(1, N, N)
    y = []
    
    for i in range(N):
        y.append(alpha_*x[i] + beta_)

    need_y = []  
    for i in range(len(test_data)):
        need_y.append(y[n+i])
    
    return need_y 
    
'''
#------------old---------------
results_for_test_data_old = [186, 190, 193, 189]
test_data_old = [8, 9,10,11]
n_old = 7
arr_x_old = [1,2,3,4,5,6, 7]
arr_y_old = [179, 182, 180, 185, 182, 187, 190]
'''

#-----------marks-------------------
results_for_test_data_for_marks = [87,98,55,66] ###!!!!!!!
test_data_for_marks = [31,32,33,34]
n_for_marks = 30
arr_x_for_marks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
arr_y_for_marks = [90,56,80,72,95,100,54,67,72,89,74,59,63,81,93,51,65,72,87,99,52,83,67,54,86,92,56,78,54,95] ###!!!!!
alpha = regression(arr_x_for_marks, arr_y_for_marks, n_for_marks)[0]
beta = regression(arr_x_for_marks, arr_y_for_marks, n_for_marks)[1]

plot_graph2("x", "y", "linear regression", alpha, beta, test_data_for_marks, results_for_test_data_for_marks,n_for_marks,arr_x_for_marks,arr_y_for_marks)

print(evaluate(test_data_for_marks, alpha, beta,results_for_test_data_for_marks,n_for_marks))

'''
#-----------years-------------------
results_for_test_data_for_years = []
test_data_for_years= [14,15,16]
n_for_years = 13
arr_x_for_years = [1,2,3,4,5,6,7,8,9,10,11,12,13]
arr_y_for_years = [154.9, 181.825, 187.525, 181.175, 175, 173.775, 173.60, 179.931, 182.172, 178.194,181.134,186.660, 181.0]

alpha = regression(arr_x_for_years, arr_y_for_years, n_for_years)[0]
beta = regression(arr_x_for_years, arr_y_for_years, n_for_years)[1]

plot_graph2("x", "y", "linear regression", alpha, beta, test_data_for_years, results_for_test_data_for_years,n_for_years,arr_x_for_years,arr_y_for_years)
#plot_graph("x", "y", "linear regression", alpha, beta,n_for_years,arr_x_for_years,arr_y_for_years)
#print(evaluate(test_data_for_years, alpha, beta,results_for_test_data_for_years,n_for_years))
'''
