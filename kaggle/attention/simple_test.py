import numpy as np
import data_utility as du

for i,j in enumerate(du.test_loader):
    print(i)
    print(j)
    if i ==10:
        exit()