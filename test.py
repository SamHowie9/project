import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# A = [[[[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]]],
#      [[[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]]],
#      [[[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]]]]

A = [[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
     [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
     [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]


print(np.array(A).shape)

print(max(max(max(row) for row in A)))

print(np.max(A))

# print(np.array(A)/max(max(max(row) for row in A)))
print(np.array(A)/np.max(A).T)

plt.imshow((np.array(A)/max(max(max(row) for row in A))).T)

plt.show()