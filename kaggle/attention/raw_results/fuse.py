import pandas as pd
import os
import numpy as np
import pandas as pd
import csv
import os
from functools import reduce
from scipy import stats
from statistics import mode
import collections
import stringdist
import textdistance
# stringdist.levenshtein




def walkfiles(entreis):
    l = []
    # print(entreis)
    id_arr = pd.read_csv(entreis[0])['Id']
    for i in sorted(entreis):
        print(i)
        l.append(np.array(pd.read_csv(i)['Predicted']))
    return np.array(l), id_arr


def fusion(bigarr):
    rows, cols = bigarr.shape[0], bigarr.shape[1]
    result = []
    for i in range(rows):
        # print(bigarr[i])
        # print(stats.mode(bigarr[i]))
        m, _ = stats.mode(bigarr[i])
        result.append(m[0])
        # print(m)
        # exit(1)
    return result

if __name__=="__main__":
    files = []
    root = "/Users/robert/Documents/CMU/19Fall/11785/11785-deep-learning/kaggle/attention/raw_results/"
    for entry in os.listdir(root):
        if entry.endswith(".csv"):
            files.append(str(root + str(entry)))
    bigarr, id_arr = walkfiles(files)
    bigarr = bigarr.T
    n = 0
    result = []
    for arr in bigarr:
        # print(arr)
        n += 1
        print("sequence:{}".format(n))
        distance = []
        for j in range(len(arr)):
            dist = 0.0
            for k in range(len(arr)):
                dist += stringdist.levenshtein(arr[j], arr[k])
            distance.append(dist)
        cur_ans = arr[distance.index(min(distance))]
        print("Chose: {}".format(cur_ans))
        result.append(cur_ans)

with open('hw4p2_submission_esp.csv', 'w+') as f:
    f.write('Id,Predicted\n')
    for i in range(len(result)):
        f.write(str(i) + ',' + str(result[i]) + '\n')

