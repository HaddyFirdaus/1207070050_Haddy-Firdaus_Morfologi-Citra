import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Reading the input image
img = cv2.imread('struktur.png', 0)

# Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)

img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)

fig, axes = plt.subplots(3, 2, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(img, cmap = 'gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins = 256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_erosion, cmap = 'gray')
ax[2].set_title("Citra Output Erosi")
ax[3].hist(img_erosion.ravel(), bins = 256)
ax[3].set_title("Histogram Citra Output Erosi")

ax[4].imshow(img_dilation, cmap = 'gray')
ax[4].set_title("Citra Output Dilasi")
ax[5].hist(img_dilation.ravel(), bins = 256)
ax[5].set_title("Histogram Citra Output Erosi")

plt.show()