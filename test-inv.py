import numpy as np


x = 1.0
s2n = 5
sigma_x = x/s2n

n = 1_000_000
xvals = np.random.normal(loc=x, scale=sigma_x, size=n)

xinv = 1/xvals

mean = xinv.mean()

print('mean:', mean)
print('bias:', mean/(1/x) - 1)
print('expected:', 1/(1 - 1/s2n**2) - 1)

