import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
# import cv2



image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_54975668.png")

print(image.shape)

# print(np.min(image[0]), np.max(image[0]))
# print(np.min(image[1]), np.max(image[1]))
# print(np.min(image[2]), np.max(image[2]))
# print()






# # A = [[[[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]]],
# #      [[[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]]],
# #      [[[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]]]]
#
# A = [[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
#      [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
#      [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]
#
#
# print(pd.DataFrame(A[0]))
#
#
# print(np.array(A).shape)
#
# print(max(max(max(row) for row in A)))
#
# print(np.max(A))
#
# # print(np.array(A)/max(max(max(row) for row in A)))
# print(np.array(A)/np.max(A).T)
#
# B = (np.array(A)/max(max(max(row) for row in A))).T
#
# C = cv2.resize(B, (10, 10))
#
# plt.imshow(C)
#
# # plt.imshow((np.array(A)/max(max(max(row) for row in A))).T)
#
# plt.show()