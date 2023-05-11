import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.datasets import load_svmlight_file


from main import Gd, Adgd, Bb, Nesterov,Armijo
from loss_functions import logistic_loss, logistic_gradient
data = pd.read_csv("./datasets/Skin_NonSkin.txt", sep="\s+", header=None, skiprows=1, names = ["B", "G", "R", "Skin_NonSkin"])

data['Skin_NonSkin'] = data['Skin_NonSkin'].map(lambda x: int(x-1)) #Transform the label to 0,1 since the loss function only supports labels from {0, 1}

X= data.drop('Skin_NonSkin', axis=1).values
y = data['Skin_NonSkin'].values
it_max = 50
n, d = X.shape

def logistic_smoothness(X):
    return 0.25 * np.max(la.eigvalsh(X.T @ X / X.shape[0]))

L = logistic_smoothness(X)
l2 = L / n
w0 = np.zeros(d)

def loss_func(w):
    return logistic_loss(w, X, y, l2)


def grad_func(w):
    return logistic_gradient(w, X, y, l2)



gd = Gd(lr=1 / L, loss_func=loss_func, grad_func=grad_func, it_max=it_max)
gd.run(w0=w0)

adgd = Adgd(loss_func=loss_func, grad_func=grad_func, eps=0, it_max=it_max)
adgd.run(w0=w0)

bb1 = Bb(loss_func=loss_func, grad_func=grad_func, option='1', it_max=it_max)
bb1.run(w0=w0)

bb2 = Bb(loss_func=loss_func, grad_func=grad_func, option='2', it_max=it_max)
bb2.run(w0=w0)

nest = Nesterov(lr=1 / L, loss_func=loss_func, grad_func=grad_func, it_max=it_max)
nest.run(w0=w0)

armijo=Armijo(lr0=1 / L, loss_func=loss_func, grad_func=grad_func, it_max=it_max)
armijo.run(w0=w0)

optimizers = [gd,adgd,nest,armijo]
markers = ['o', '*',"x","^"]

for opt, marker in zip(optimizers, markers):
    opt.compute_loss_on_iterates()
f_star = np.min([np.min(opt.losses) for opt in optimizers])

plt.figure(figsize=(8, 6))
labels = ['GD','AdGD','Nest','Armijo']
for opt, marker, label in zip(optimizers, markers + ['.', 'X'], labels):
    opt.plot_losses(marker=marker, f_star=f_star, label=label)
# plt.yscale('log') logaritmic scale
plt.xlabel('Iteration')
plt.ylabel(r'$f(x^k) - f_*$')
plt.legend()
plt.tight_layout()
plt.show()




w1=adgd.w
w2=gd.w
w3=bb1.w
w4=bb2.w
w5=nest.w
w6=armijo.w
print(w1)
print(w2)
print(w3)
print(w4)
print(w5)
print(w6)
