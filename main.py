import numpy as np
import cv2
import matplotlib.pyplot as plt
from path import path

# Read image 'image1.jpg' as grayscale
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

_, img_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

img_inverted = 255 - img

# Equalizar o histograma
img_eq = cv2.equalizeHist(img)

# Mostrar antes e depois
plt.figure(figsize=(10, 5))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(img_eq, cmap='gray')
plt.title('Imagem Equalizada')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(img_thresh, cmap='gray')
plt.title('Imagem Binarizada')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(img_inverted, cmap='gray')
plt.title('Imagem Invertida')
plt.axis('off')

plt.show()