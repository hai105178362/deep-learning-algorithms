import numpy as np
mu, beta = 250, 50 # location and scale
s = np.random.gumbel(mu, beta)
print(s)