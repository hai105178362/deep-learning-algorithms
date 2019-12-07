import numpy as np
import data_utility as du

mu, beta = 250, 50  # location and scale
s = np.random.gumbel(mu, beta)
print(s)

print(du.letter_list[18, 5, 32, 20, 8, 5, 32, 19, 1, 20])
