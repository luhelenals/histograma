import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os

def plot_image_and_histogram(img, img_eq, img_inverted):
    # Calcular os histogramas
    hist_img = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_img_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])
    hist_img_inverted = cv2.calcHist([img_inverted], [0], None, [256], [0, 256])

    # Criar a figura para exibir as imagens e seus histogramas
    plt.figure(figsize=(18, 12))

    # Mostrar imagem original
    plt.subplot(3, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem Original')
    plt.axis('off')

    # Histograma da imagem original
    plt.subplot(3, 2, 2)
    plt.plot(hist_img, color='black')
    plt.title('Histograma - Imagem Original')
    plt.xlim([0, 255])
    plt.gca().set_ylim(bottom=0)

    # Mostrar imagem equalizada
    plt.subplot(3, 2, 3)
    plt.imshow(img_eq, cmap='gray')
    plt.title('Imagem Equalizada')
    plt.axis('off')

    # Histograma da imagem equalizada
    plt.subplot(3, 2, 4)
    plt.plot(hist_img_eq, color='blue')
    plt.title('Histograma - Imagem Equalizada')
    plt.xlim([0, 255])
    plt.gca().set_ylim(bottom=0)

    # Mostrar imagem invertida
    plt.subplot(3, 2, 5)
    plt.imshow(img_inverted, cmap='gray')
    plt.title('Imagem Invertida')
    plt.axis('off')

    # Histograma da imagem invertida
    plt.subplot(3, 2, 6)
    plt.plot(hist_img_inverted, color='red')
    plt.title('Histograma - Imagem Invertida')
    plt.xlim([0, 255])
    plt.gca().set_ylim(bottom=0)

    # Ajustar o layout
    plt.tight_layout()
    plt.show()

def process_image(file):
    # Ler imagem em escala de cinza
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_inverted = 255 - img
    img_eq = cv2.equalizeHist(img)

    return img, img_eq, img_inverted

def main():
    # Solicitar o nome do arquivo ao usuário
    image_name = input("Digite o nome da imagem (ou pressione Enter para processar todas as imagens): ").strip()

    # Verificar se o nome do arquivo foi inserido
    if image_name:
        image_path = os.path.join('./images', image_name)

        if os.path.exists(image_path):
            # Processar a imagem especificada
            img, img_eq, img_inverted = process_image(image_path)

            # Exibir a imagem e seus histogramas
            plot_image_and_histogram(img, img_eq, img_inverted)
        else:
            print("Imagem não encontrada no diretório './images'.")
    else:
        # Processar todas as imagens na pasta
        for file in glob.glob('./images/*'):
            img, img_eq, img_inverted = process_image(file)

            # Exibir a imagem e seus histogramas para cada imagem
            plot_image_and_histogram(img, img_eq, img_inverted)

if __name__ == '__main__':
    main()
