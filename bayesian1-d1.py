import pandas as pd
import matplotlib.pyplot as plt
import pylab
from pandas import DataFrame
import statistics
import scipy.stats

from sklearn.model_selection import train_test_split

df=pd.read_csv("sobar-72.csv")
features=df.keys().to_list()
filterd=features[1:7]+features[8:19]
print(filterd)
print(len(filterd))
X = df[filterd]
y=df.ca_cervix
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
lis=df.keys().to_list()
print(lis)
index1=[]
index2=[]

print(df[lis[19]])

for i in range(len(df[lis[19]].to_list())):
    if(df[lis[19]][i]==1):
        index1.append(i)

    else:
        index2.append(i)

prior1=len(index1)/(len(index2)+len(index1))
prior2=len(index2)/(len(index2)+len(index1))
print(prior1)
print(prior2)

mean1=[]
sd1=[]
mean2=[]
sd2=[]
for i in filterd:
    flis=X_train[i].index.to_list()
    lis1=[]
    lis2=[]

    for j in range(len(flis)):
        if(flis[j] in index1):
            lis1.append(X_train[i][flis[j]])
        else:
            lis2.append(X_train[i][flis[j]])
    mean1.append(statistics.mean(lis1))
    mean2.append(statistics.mean(lis2))
    sd1.append(statistics.stdev(lis1))
    sd2.append(statistics.stdev(lis2))
print(sd1)
print(sd2)
Accuracy=[]
print(X_test[filterd[0]][19])

for i in range(len(filterd)):

    flis1=X_test[filterd[i]].index.to_list()
    count=0
    g1=scipy.stats.norm(mean1[i],sd1[i])
    g2 = scipy.stats.norm(mean2[i], sd2[i])

    for j in range(len(flis1)):

        x1=g1.pdf(X_test[filterd[i]][flis1[j]])*prior1
        x2 = g2.pdf(X_test[filterd[i]][flis1[j]])*prior2
        if(x1>x2 and y_test[flis1[j]]==1):
            count=count+1
        elif(x1<=x2 and y_test[flis1[j]]==0):
            count=count+1
        else:
            continue
    Accuracy.append((count/len(flis1))*100)
for i in range(len(Accuracy)):
    print("Accurary for feature   "+ str(i+1)+"   " + str(Accuracy[i]))
#combining Classifiers
print(X_test)
flis2 = X_test[filterd[0]].index.to_list()
print(flis2)
count2=0
for i in range(len(X_test[filterd[0]].index.to_list())):

    lis=[]
    for j in range(len(filterd)):
        lis.append(df[filterd[j]][flis2[i]])
    voting=[]

    for k in range(len(lis)):
        g1 = scipy.stats.norm(mean1[k], sd1[k])
        g2 = scipy.stats.norm(mean2[k], sd2[k])
        x1 = g1.pdf(lis[k]) * prior1
        x2 = g2.pdf(lis[k]) * prior2
        if (x1 > x2):
            voting.append(1)
        else:
            voting.append(0)
    num1=voting.count(1)
    num2=voting.count(0)
    if(num1>num2):
        result=1
    else:
        result=0
    if ( result==y_test[flis2[i]]):
        count2 = count2 + 1

    else:
        continue
print("combined Accuracy is   "+str((count2/len(flis2))*100)+"%")
