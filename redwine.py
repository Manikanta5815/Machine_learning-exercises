import pandas as pd
import matplotlib.pyplot as plt
import pylab
from pandas import DataFrame
import statistics
import scipy.stats

from sklearn.model_selection import train_test_split

df=pd.read_csv("winequality-red.csv")
keysstr=df.keys().to_list()[0]
keys=keysstr.split(";")
features=keys[:-1]
label=keys[len(keys)-1]
print(features)
print(label)
print(df)
print(df[keysstr][0])
lis=[]
for i in range(1599):
    lis.append(df[keysstr][i].split(";"))
cols=[]
dict={}
for i in range(len(keys)):
    lis1=[]
    for j in range(1599):
        lis1.append(lis[j][i])
    dict[keys[i]]=lis1

data=pd.DataFrame.from_dict(dict)
features1=data.keys().to_list()
print(data)
X = data[features[:]]

y=data.quality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
import numpy as np
from sklearn.linear_model import LinearRegression
models=[]
results=[]
target=y.to_list()
target1=[float(numeric_string) for numeric_string in target]

mod=LinearRegression()
for i in range(len(features)):
    regmodel = LinearRegression()
    regmodel.fit(pd.DataFrame(X_train[features[i]]),y_train)
    result=list(regmodel.predict(pd.DataFrame(X_test[features[i]])))
    result1=[round(num) for num in result]
    results.append(result)
print(results)
means=[]
for i in range(len(results[0])):
    lis=[]
    for j in range(len(features)):
        lis.append(results[j][i])
    means.append((statistics.mean(lis)))
s=0
y_tes1=list(y_test)
y_tes=[float(numeric_string) for numeric_string in y_tes1]
errors=[]
for i in range(len(y_tes)):
    errors.append((means[i]-y_tes[i])*(means[i]-y_tes[i]))
    s+=(means[i]-y_tes[i])*(means[i]-y_tes[i])
rmse=(s/len(y_tes))**0.5
print(rmse)
print(max(errors))
print(min(errors))

from sklearn.preprocessing import PolynomialFeatures
import operator

import numpy as np
import matplotlib.pyplot as plt

polynomial_features = PolynomialFeatures(degree=2)
model = LinearRegression()
resultsfina1=[]
for i in range(len(features)):
    x = X_train[features[i]][:, np.newaxis]
    y = y_train[:, np.newaxis]
    x_poly = polynomial_features.fit_transform(x)


    model.fit(x_poly, y)
    xtest=X_test[features[i]][:,np.newaxis]
    xtest1=polynomial_features.fit_transform(xtest)
    result=list(model.predict(xtest1))
    result1=[]
    for i in range(len(result)):
        result1.append((result[i]))
    resultsfina1.append(result1)
    print(result1)
