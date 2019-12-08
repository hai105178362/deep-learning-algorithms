import numpy as np
import data_utility as du

for i, j in enumerate(du.train_loader):
    for num in (j[1]).detach().cpu().numpy():
        cur = [(i - 1) for i in num if i != 0]
        # print(cur)
        print([du.letter_list[i] for i in cur])
