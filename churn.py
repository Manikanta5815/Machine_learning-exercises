import pandas as pd
data=open("E:\data Science\churn.txt")
lines=data.readlines()
line=[]
for i in range(len(lines)):
    line.append(lines[i].split("\t"))
features=(line[0][0].split("    "))
or_data=[]
for i in range(len(line)-1):
    or_data.append(line[i+1])
fin_data=[]
cleaned=[]
for i in range(len(or_data)):
    if("" not in or_data[i]):

        cleaned.append(or_data[i])
    else:
        continue
print(len(or_data))
print(len(cleaned))
