import pandas as pd
import matplotlib.pyplot as plt
import pylab
from pandas import DataFrame
import statistics
import scipy.stats

from sklearn.model_selection import train_test_split

df=pd.read_csv("winequality-white.csv")
keysstr=df.keys().to_list()[0]
keys=keysstr.split(";")
features=keys[:-1]
label=keys[len(keys)-1]
print(features)
print(label)
print(df)
print(df[keysstr][0])
lis=[]
for i in range(4898):
    lis.append(df[keysstr][i].split(";"))
cols=[]
dict={}
for i in range(len(keys)):
    lis1=[]
    for j in range(4898):
        lis1.append(lis[j][i])
    dict[keys[i]]=lis1

data=pd.DataFrame.from_dict(dict)
features1=data.keys().to_list()
print(data)
X = data[features[:]]
lis=[2,3,4,5,6,7,8]
y=data.quality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
accuracy={}

import statistics
for i in range(len(lis)):

        clf = MLPClassifier(random_state=1, hidden_layer_sizes=(lis[i], ),max_iter=500, activation='logistic').fit(X_train, y_train)

        scores = cross_val_score(clf, X_test, y_test, cv=5)

        accuracy[lis[i]]=(1-statistics.mean(scores))
print(accuracy)
plt.plot(*zip(*sorted(accuracy.items())))
plt.title("whitewine",color="brown")
plt.xlabel("hidden neurons",color="blue")
plt.ylabel("error rate",color="red")
plt.show()
