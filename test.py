import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
# import cv2


A = [1, 2, 3, 4, 5, 6, 7, 8, 9]

print(A[:3])

# print(A[:-3])
# print(A[-3:])



# extracted_features = np.load("Variational Eagle/Extracted Features/Normalised to r/15_feature_300_epoch_features_1.npy")
#
# print(extracted_features.shape)
# print(extracted_features[0].shape)
# print(extracted_features[0][100].shape)
# print(extracted_features[0][100][7])





# A = []
#
# image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_54975668.png")
#
# A.append(image)
# A.append(image)
# A.append(image)
# A.append(image)
#
# print(np.array(A).shape)
# print(image.T[0].shape)
#
# print(np.min(image.T[0]), np.max(image.T[0]))
# print(np.min(image.T[1]), np.max(image.T[1]))
# print(np.min(image.T[2]), np.max(image.T[2]))
#
# # print(np.min(image[0]), np.max(image[0]))
# # print(np.min(image[1]), np.max(image[1]))
# # print(np.min(image[2]), np.max(image[2]))
# # print()






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