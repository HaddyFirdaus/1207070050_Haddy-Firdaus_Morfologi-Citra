import cv2
import numpy as np
from skimage import data
from skimage.io import imread

import matplotlib.pyplot as plt
#%matplotlib inline

#image = data.retina()
#image = data.astronaut()
image = imread(fname="aqua2.jpg")

print(image.shape)
plt.imshow(image)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# defining the range of masking
blue1 = np.array([110, 50, 50])
blue2 = np.array([130, 255, 255])

# initializing the mask to be
# convoluted over input image
mask = cv2.inRange(hsv, blue1, blue2)

# passing the bitwise_and over
# each pixel convoluted
res = cv2.bitwise_and(image, image, mask=mask)

# defining the kernel i.e. Structuring element
kernel = np.ones((5, 5), np.uint8)

# defining the opening function
# over the image and structuring element
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

fig, axes = plt.subplots(1, 2, figsize=(12, 12))
ax = axes.ravel()

ax[0].imshow(mask)
ax[0].set_title("Citra Input 1")

ax[1].imshow(opening, cmap='gray')
ax[1].set_title('Citra Input 2')

plt.show()

import cv2
import numpy as np


def process_webcam():
    # Menginisialisasi video capture dari webcam pertama pada komputer
    screenRead = cv2.VideoCapture(0)

    # Melakukan loop jika pengambilan gambar telah diinisialisasi
    while True:
        # Membaca frame dari kamera
        ret, image = screenRead.read()

        # Mengubah ruang warna BGR ke HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Menentukan rentang warna untuk masking (sesuaikan dengan kondisi pencahayaan)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([150, 255, 255])

        # Membuat mask menggunakan rentang yang ditentukan
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Menggunakan operasi bitwise_and pada setiap piksel
        res = cv2.bitwise_and(image, image, mask=mask)

        # Membentuk kernel (structuring element)
        kernel = np.ones((5, 5), np.uint8)

        # Menggunakan operasi morphological opening pada gambar menggunakan kernel yang telah dibentuk
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Menampilkan mask dan hasil operasi opening di jendela
        cv2.imshow('Mask', mask)
        cv2.imshow('Opening', opening)

        # Menunggu tombol 'a' untuk menghentikan program
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

    # Membebaskan memori yang digunakan
    cv2.destroyAllWindows()

    # Menutup jendela dan melepaskan webcam
    screenRead.release()


# Memanggil fungsi untuk menjalankan webcam
process_webcam()


