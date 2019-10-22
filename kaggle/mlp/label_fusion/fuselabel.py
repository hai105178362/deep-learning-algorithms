import pandas as pd
import csv

a = pd.read_csv("25ep_test_61.csv")['label']
b = pd.read_csv("59.csv")['label']
c = pd.read_csv("609.csv")['label']

newtab = []
for i, j, k in zip(a, b, c):
    if i == j or i == k or j == k:
        if i == j:
            newtab.append(i)
        elif i == k:
            newtab.append(k)
        else:
            newtab.append((j))
    else:
        newtab.append(i)

print(len(newtab))
with open('finalresult.csv', mode='w') as csv_file:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(newtab)):
        writer.writerow({'id': i, 'label': int(newtab[i])})
