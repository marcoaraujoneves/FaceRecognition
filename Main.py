import os
import numpy as np
from PIL import Image

listaImagens = np.empty((10, 3, 100, 100), dtype="float32")

nomeImagens = os.listdir("./very-easy")

for i in range(len(nomeImagens)):
    imagem = Image.open("./very-easy/" + nomeImagens[i])
    arrayImagem = np.asarray(imagem, dtype="float32")
    listaImagens[i, :, :, :] = arrayImagem.transpose(2, 1, 0)

