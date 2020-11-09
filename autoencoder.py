import pandas as pd
import matplotlib.pyplot as plt
import pylab

data = pd.read_csv('data_banknote_authentication.txt', sep=",", header=None)
data.columns = ['variance','skew','curtosis','entropy','class']

X_train=data[data['class']==0]

X_test=data[data['class']==1]
from sknn import ae, mlp