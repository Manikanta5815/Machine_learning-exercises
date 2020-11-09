import pandas as pd
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
print (df)
feature_cols = ['c1','c2','c3','c4','c5','c6','c7']
X = df[feature_cols]
y = df.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(X_train)
labels=[1,2,3]
index1=[]
index2=[]
index3=[]
lis=df.keys().to_list()
print(df[lis[7]])
for i in range(len(df[lis[7]].to_list())):
    if(df[lis[7]][i]=="1"):
        index1.append(i)
    elif(df[lis[7]][i]=="2"):
        index2.append(i)
    else:
        index3.append(i)

for i in feature_cols:
    flis=X_train[i].index.to_list()
    lis1=[]
    lis2=[]
    lis3=[]
    for j in range(len(flis)):
        if(flis[j] in index1):
            lis1.append(X_train[i][flis[j]])
        elif (flis[j] in index2):
            lis2.append(X_train[i][flis[j]])
        else:
            lis3.append(X_train[i][flis[j]])
    plt.subplot(1,3,1)
    plt.hist(lis1)
    plt.subplot(1, 3, 2)
    plt.hist(lis2)
    plt.subplot(1, 3, 3)
    plt.hist(lis3)
    plt.show()
