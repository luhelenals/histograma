import cv2
import matplotlib.pyplot as plt
from path import image1, image2, image3, image4

# Ler imagem em escala de cinza
img = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)

# Inverter imagem
img_inverted = 255 - img

# Equalizar o histograma
img_eq = cv2.equalizeHist(img)

# Calcular os histogramas
hist_img = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_img_inverted = cv2.calcHist([img_inverted], [0], None, [256], [0, 256])
hist_img_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

# Plotar os histogramas
plt.figure(figsize=(12, 6))

# Histograma da imagem original
plt.subplot(1, 3, 1)
plt.plot(hist_img, color='black')
plt.title('Histograma - Imagem Original')
plt.xlim([0, 255])  # Limitar o eixo x para garantir que comece em 0
plt.ylim([0, max(hist_img) * 1.05])  # Ajustar o eixo y para não cortar o gráfico
plt.gca().set_ylim(bottom=0)  # Garantir que o y comece em 0

# Histograma da imagem invertida
plt.subplot(1, 3, 2)
plt.plot(hist_img_inverted, color='red')
plt.title('Histograma - Imagem Invertida')
plt.xlim([0, 255])
plt.ylim([0, max(hist_img_inverted) * 1.05])
plt.gca().set_ylim(bottom=0)

# Histograma da imagem equalizada
plt.subplot(1, 3, 3)
plt.plot(hist_img_eq, color='blue')
plt.title('Histograma - Imagem Equalizada')
plt.xlim([0, 255])
plt.ylim([0, max(hist_img_eq) * 1.05])
plt.gca().set_ylim(bottom=0)

# Mostrar os gráficos
plt.tight_layout()
plt.show()
