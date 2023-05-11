import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.calibration import calibration_curve
from scipy.stats import logistic
from math import exp
import matplotlib.pyplot as plt

def sigmoid(x):
    "Numerically-stable and fast sigmoid function."
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        z = exp(x)
        return z / (1 + z)



w_adgd=[ 0.02784233 , 0.00990059 ,-0.02659189]
w_gd=[ 0.02200344 , 0.01018997, -0.02227494]
w_nest=[ 0.02880746 , 0.00937325, -0.02687326]
data = pd.read_csv("./datasets/Skin_NonSkin.txt", sep="\s+", header=None, skiprows=1, names = ["B", "G", "R", "Skin_NonSkin"])

data['Skin_NonSkin'] = data['Skin_NonSkin'].map(lambda x: int(x-1)) #Transform the label to 0,1 since the loss function only supports labels from {0, 1}

X= data.drop('Skin_NonSkin', axis=1).values
y = data['Skin_NonSkin'].values
it_max = 1000
n, d = X.shape
z1= np.dot(X, w_adgd)
z2=np.dot(X, w_gd)
z3=np.dot(X,w_nest)

sigmoidvector= np.vectorize(sigmoid)
result_1=sigmoidvector(z1)
result_2=sigmoidvector(z2)
result_3=sigmoidvector(z3)

prob_true, prob_pred = calibration_curve(y, result_1, n_bins=5)
print(prob_true)
print(prob_pred)

losses1 = np.subtract(y, result_1)**2
brier_score1 = losses1.sum()/n

losses2 = np.subtract(y, result_2)**2
brier_score2 = losses2.sum()/n

losses3 = np.subtract(y, result_3)**2
brier_score3 = losses3.sum()/n

fpr1, tpr1, _ = metrics.roc_curve(y,result_1)
auc1 = metrics.roc_auc_score(y, result_1)

#create ROC curve
plt.plot(fpr1,tpr1,label="AUC="+str(auc1))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


fpr2, tpr2, _ = metrics.roc_curve(y,result_2)
auc2 = metrics.roc_auc_score(y, result_2)

#create ROC curve
plt.plot(fpr2,tpr2,label="AUC="+str(auc2))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

plt.figure(0).clf()
plt.plot(fpr1,tpr1,label="ADGD, AUC="+str(auc1))
plt.plot(fpr2,tpr2,label="GD, AUC="+str(auc2))
plt.legend(loc=4)
plt.show()