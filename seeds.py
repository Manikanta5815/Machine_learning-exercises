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
# Splitting the data line Wise
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
clf = DecisionTreeClassifier(criterion="entropy")
clf1 = DecisionTreeClassifier(criterion="gini")

clf = clf.fit(X_train,y_train)
clf1=clf1.fit(X_train,y_train)
y_pred1 = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))

y_pred2 = clf1.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))
from sklearn import tree

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=500)
tree.plot_tree(clf,
               feature_names = feature_cols,
               class_names=['1','2','3'],
               filled = True)
fig.savefig('imagename.png')
tree.plot_tree(clf1,
               feature_names = feature_cols,
               class_names=['1','2','3'],
               filled = True)
fig.savefig('imagename1.png')