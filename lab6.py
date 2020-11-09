import pandas as pd
import matplotlib.pyplot as plt
import pylab
from pandas import DataFrame
import statistics
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import sklearn.feature_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv("turkiye-student-evaluation_generic.csv")
features=df.keys().to_list()
print(features)

len=len(df[features[0]].to_list())
lis=[]
for i in range(len):
    lis.append(str(df[features[0]][i])+str(df[features[1]][i]))
df['class_label']=lis
df=df.drop([features[0],features[1]],axis=1)
X = df[features[2:]]
y=df.class_label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier(criterion="entropy",max_depth=3)

clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("error1",1-metrics.accuracy_score(y_test, y_pred))
features1=features[5:]


t=[0.02,0.03,0.0025,0.004,0.032]
accuracy={}
mi = []
knnaccuracy=[]
for j in t:
    t_features = []
    for i in features1:
        x = pd.DataFrame(X_train[i])
        mi1 = sklearn.feature_selection.mutual_info_classif(x, y_train)[0]
        mi.append(mi1)
        if (mi1 > j):
            t_features.append(i)
    print(t_features)
    X1 = df[t_features]

    y1 = df.class_label
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3)
    clf1 = DecisionTreeClassifier(criterion="entropy",max_depth=3)
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training sets
    knn.fit(X_train1, y_train1)

    # Predict the response for test dataset
    y_pred2 = knn.predict(X_test1)
    knnaccuracy.append(1 - metrics.accuracy_score(y_test1, y_pred2))

    clf1 = clf1.fit(X_train1, y_train1)
    y_pred1 = clf1.predict(X_test1)
    accuracy[j]=1 - metrics.accuracy_score(y_test1, y_pred1)
print(knnaccuracy)

print("error2")
print(accuracy)
plt.plot(*zip(*sorted(accuracy.items())))
plt.xlabel("threshold")
plt.ylabel("error")
plt.show()
