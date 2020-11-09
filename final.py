import pandas as pd
import matplotlib.pyplot as plt
import pylab
file=open("chronic_kidney_disease.ARFF","r")
lines=file.readlines()
k=lines.index("@data\n")
print(lines)
print(k)
final_lis=lines[k+1::]
data=[]
labels=["Age","blood_pressure","specific gravity","albumin","sugar"]

for i in range(len(final_lis)-1):
    lis=[]
    str=final_lis[i]

    spl=str.split(",")
    for h in range(5):
        lis.append(spl[h])
    lis.append((spl[len(spl)-1]))
    data.append(lis)
print(data)
cleaned=[]
for i in range(len(data)):
    if("?" not in data[i]):

        cleaned.append(data[i])
    else:
        continue
print(len(cleaned))

for a in range(5):
    for b in range(a+1,5):
        data1 = []
        data2 = []
        data3 = []
        data4 = []
        for i in range(len(cleaned)):
            if (data[i][5] == "ckd\n"):
                data1.append(float(cleaned[i][a]))
                data2.append(float(cleaned[i][b]))
            else:
                data3.append(float(cleaned[i][a]))
                data4.append(float(cleaned[i][b]))
        plt.scatter(data1, data2, c='coral')
        plt.scatter(data3, data4, c='lightblue')
        plt.xticks(rotation=45)
        plt.xlabel(labels[a])
        plt.ylabel(labels[b])
        plt.show()


