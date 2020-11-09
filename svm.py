# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
iris = datasets.load_iris()


X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)

from sklearn.svm import SVC
p=[1,2,3,4,5,8,10,12,15,18]
accuracy_fin={}

for i in p:
    c=0.01
    svm_model1 = SVC(kernel='poly', degree=i, C=0.01).fit(X_train, y_train)
    svm_predictions1 = svm_model1.predict(X_test)

    accuracy1 = metrics.accuracy_score(svm_predictions1, y_test)
    svm_model2 = SVC(kernel='poly', degree=i, C=0.1).fit(X_train, y_train)
    svm_predictions2 = svm_model2.predict(X_test)

    accuracy2 = metrics.accuracy_score(svm_predictions2, y_test)

    if(accuracy1>accuracy2):
        accuracy=accuracy1
    else:
        accuracy=accuracy2
    k=1
    while(accuracy1>accuracy2):

        svm_model = SVC(kernel = 'poly',degree=i, C = k).fit(X_train, y_train)
        svm_predictions = svm_model.predict(X_test)

        accuracy = metrics.accuracy_score(svm_predictions,y_test)
        k=k*10
        accuracy1=accuracy2
        accuracy2=accuracy
    accuracy_fin[i]=accuracy

print(accuracy_fin)
plt.plot(*zip(*sorted(accuracy_fin.items())))
plt.title("p vs accuracy",color="brown")
plt.xlabel("p",color="blue")
plt.ylabel("accuracy",color="red")
plt.show()
