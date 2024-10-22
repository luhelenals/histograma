import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

# Array de tuples com as imagens editadas
imageArr = []

# Read image 'image1.jpg' as grayscale
path = 'C:/Users/ALUNO SESI E SENAI.MATRIZ/Desktop/Luiza/images/*'
for file in glob.glob(path):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    img_inverted = 255 - img

    # Equalizar o histograma
    img_eq = cv2.equalizeHist(img)

    imageArr.append((img, img_eq, img_inverted))

# Criar a figura para exibir as imagens
plt.figure(figsize=(15, 10))  # Ajuste o tamanho conforme necessário

# Loop para exibir as imagens e suas edições
plt.subplot(len(imageArr), 3, 1)
plt.title('Imagem Original')
plt.subplot(len(imageArr), 3, 2)
plt.title('Imagem Equalizada')
plt.subplot(len(imageArr), 3, 3)
plt.title('Imagem Invertida')

for i, tup in enumerate(imageArr):
    img, img_eq, img_inverted = tup

    # Mostrar original
    plt.subplot(len(imageArr), 3, 3 * i + 1)
    plt.imshow(img, cmap='gray')
    
    plt.axis('off')

    # Mostrar equalizada
    plt.subplot(len(imageArr), 3, 3 * i + 2)
    plt.imshow(img_eq, cmap='gray')
    
    plt.axis('off')

    # Mostrar invertida
    plt.subplot(len(imageArr), 3, 3 * i + 3)
    plt.imshow(img_inverted, cmap='gray')
    
    plt.axis('off')

# Exibir todas as imagens
plt.tight_layout()  # Melhora o espaçamento entre os subplots
plt.show()