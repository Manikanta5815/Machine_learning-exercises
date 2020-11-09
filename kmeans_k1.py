import pandas as pd
import matplotlib.pyplot as plt
import pylab
from pandas import DataFrame
import statistics
import scipy.stats

from sklearn.model_selection import train_test_split

df=pd.read_csv("heart_failure_clinical_records_dataset.csv")
features=df.keys().to_list()
print(features)
X=df[features[:-1]]
print(X)
sse = {}
from sklearn.cluster import KMeans
import centroid_initialization as cent_init
for k in range(1, 10):
    random_c=cent_init.random(X,12)
    print(random_c)