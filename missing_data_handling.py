import matplotlib.pyplot as plt
import pylab
from pandas import DataFrame

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
file=open("seeds_dataset.txt","r")
lines=file.readlines()
print(lines)
line=[]
# Splitting the data line wise
for i in range(len(lines)):
    line.append(lines[i].split("\t"))
print(line)
# Getting Numeric value from String Value
for i in range(len(line)):
    for j in range(len(line[i])-1):
        line[i][j]=float(line[i][j])
for i in range(len(line)):

        line[i][7]=line[i][7][0]
# Creating The DataFrame
df = DataFrame (line,columns=['c1','c2','c3','c4','c5','c6','c7','label'])

feature_cols = ['c1','c2','c3','c4','c5','c6','c7']
index=df.index.to_list()
X = df[feature_cols]
y = df.label
S_complete,missing = train_test_split(df, test_size=0.1, shuffle=True)
missing_index=missing.index.to_list()
missing_copy=missing.copy()
print(missing_index)
import random
missingnums=[]
initiallist=[]
finallist=[]
import numpy as np

for i in range(len(missing_index)):
    list = missing.loc[missing_index[i]].to_list()
    initiallist.append(list)
for i in range(len(missing_index)):
    list=missing.loc[missing_index[i]].to_list()

    num=random.sample([0,1,2,3,4,5,6],1)
    missingnums.append(num[0])
    list[num[0]]=np.nan
    finallist.append(list)
    missing.loc[missing_index[i]]=list
print(missing_copy)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=6)


imp=imputer.fit_transform(finallist)
print("values with out missing")
print(initiallist)
print("values with missing")
print(finallist)
print(imp)
error=[]
for i in range(len(finallist)):
    error.append(abs(2 * ((initiallist[i][missingnums[i]] - imp[i][missingnums[i]]) / (
                initiallist[i][missingnums[i]] + imp[i][missingnums[i]]))))
print(error)
print("minimum error")
print(min(error))
print("maximum error")
print(max(error))
import statistics
print("mean error")
print(statistics.mean(error))
