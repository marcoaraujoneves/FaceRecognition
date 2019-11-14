import os
import numpy as np
from PIL import Image
from scipy import sum, average

_listaImagens = []
_nomeImagens = []
_normasImagens = []


def normalize(imagem):
    norm_range = imagem.max() - imagem.min()
    minimum = imagem.min()
    return (imagem - minimum)*255 / norm_range


def compare_images(imagem_1, imagem_2):
    imagem_1 = normalize(imagem_1)
    imagem_2 = normalize(imagem_2)

    difference = imagem_1 - imagem_2
    norma = sum(abs(difference))
    print(norma)


def carrega_imagens(nomePasta):
    normas_imagens = []
    nome_imagens = os.listdir("./" + nomePasta)
    lista_imagens = np.empty((len(nome_imagens), 100, 100), dtype="float32")

    for i in range(len(nome_imagens)):
        # Salvando imagem
        imagem = Image.open("./"+nomePasta+"/" + nome_imagens[i]).convert('LA')
        imagem.show()
        array_image = np.asarray(imagem, dtype="float32")
        array_image = average(array_image, -1)
        lista_imagens[i, :, :] = array_image
        normas_imagens.append(0)

        # Pegando id
        nome_imagens[i] = nome_imagens[i].split("-")[0]

    return lista_imagens, nome_imagens, normas_imagens


_listaImagens, _nomeImagens, _normasImagens = carrega_imagens("very-easy")
compare_images(_listaImagens[0], _listaImagens[1])
# for i in range(len(listaImagens)):
#    for j in range(len(listaImagens[i])):
#        for k in range(len(listaImagens[i][j])):
#            for l in range(len(listaImagens[i][j][k])):
#                normasImagens[i] += listaImagens[i][j][k][l]
#    normasImagens[i] = (normasImagens[i])**(0.5)
#    print(normasImagens[i])

# compare_images()