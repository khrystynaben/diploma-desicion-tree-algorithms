

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Припустимо, що у вас є дійсні мітки (y_true) та оцінки класифікатора (y_score)
# y_true - масив з істинними мітками (0 або 1)
# y_score - масив з оцінками, які видає класифікатор (від 0 до 1, або значення, що вказують ймовірність належності до класу 1)

def convert_to_bool(arr):
    return [1 if x == 'yes' else 0 for x in arr]



# Обчислити ROC криву та площу під нею (AUC)
def my_roc_curve(real, predicted):
    #real = convert_to_bool(real)
    #predicted = convert_to_bool(predicted)
    #print("real",real,"predicted", predicted)
    fpr, tpr, thresholds = roc_curve(real, predicted)
    roc_auc = auc(fpr, tpr)

    # Побудувати ROC криву
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()