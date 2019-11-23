import os
import random
import numpy as np
from PIL import Image
from scipy import sum, average

# Numero utilizado para comparar as normas
comparative = 99999999999


# Carrega imagens da pasta passada como argumento, separando na estrutura do dataset
def carrega_imagens(nome_pasta):
    # O dataset segue a seguinte estrutura, utilizando dicionário Python:
    # dataset = {
    #     'nome da classe (id da pessoa)': {
    #         'fotos' : ['codigo da imagem 1','codigo da imagem 2',...]
    #     }
    # }
    dataset = {}
    nome_imagens = os.listdir("./" + nome_pasta)

    for img in nome_imagens:
        id = img.split('-')[0]
        codigo = img.split('-')[1].split('.')[0]
        if not(dataset.get(id)):
            dataset.update({id: {'fotos': [codigo]}})
        else:
            dataset.get(id)["fotos"].append(codigo)

    return dataset


# A função recebe o número de imagens para testes por classe como parâmetro
def separa_conjuntos(num_teste, nome_pasta):
    dataset = carrega_imagens(nome_pasta)
    teste_set = []
    treino_set = []

    for classe in dataset:
        for i in range(num_teste):
            imagem = random.choice(dataset[classe]['fotos'])
            teste_set.append(classe + '-' + imagem)
            dataset[classe]['fotos'].remove(imagem)
        for i in dataset[classe]['fotos']:
            treino_set.append(classe + '-' + i)

    return treino_set, teste_set


# Calcula a norma da imagem
def normalize(imagem):
    norm_range = imagem.max() - imagem.min()
    minimum = imagem.min()
    return (imagem - minimum)*255 / norm_range


# Calcula a diferença entre as normas das imagens passadas como parâmetros
def compare_images(imagem_1, imagem_2):
    imagem_1 = normalize(imagem_1)
    imagem_2 = normalize(imagem_2)

    difference = imagem_1 - imagem_2
    norma = sum(abs(difference))
    return norma


# Normalizar array de diferencas
def normaliza_array(diferencas):
    maximo = diferencas[max(diferencas, key=diferencas.get)]
    minimo = diferencas[min(diferencas, key=diferencas.get)]
    amplitude = maximo - minimo

    for id in diferencas:
        diferencas[id] = (diferencas[id] - minimo) / amplitude

    return diferencas


# Realiza testes buscando a menor diferença entre as normas
def reconhecimento_norma(nome_imagem, treino_set, nome_pasta, extensao):
    imagem = np.asarray(
        Image.open("./" + nome_pasta + "/" + nome_imagem + extensao).convert('LA')
        , dtype="float32")
    imagem = average(imagem, -1)

    menor_diferenca = comparative
    output = ''
    diferencas = {}

    for img in treino_set:
        img_treino = np.asarray(
            Image.open("./" + nome_pasta + "/" + img + extensao).convert('LA')
            , dtype="float32")
        img_treino = average(img_treino, -1)

        norma = compare_images(imagem, img_treino)
        diferencas.update({img.split('-')[0]: norma})
        if norma < menor_diferenca:
            menor_diferenca = norma
            output = img.split('-')[0]

    diferencas = normaliza_array(diferencas)
    certeza = 100 - diferencas[output]
    return certeza, output, diferencas


# Realiza o teste de reconhecimento
def reconhece(nome_pasta, extensao):
    treino, teste = separa_conjuntos(1, "very-easy")

    for img in teste:
        print("-> Reconhecendo (" + img + ")")
        certeza, output, diferencas = reconhecimento_norma(img, treino, nome_pasta, extensao)
        print("\t |-> ID com melhor semelhança: " + str(output) + " , % de confiança: " + str(certeza) + "%")


reconhece('very-easy', '.jpg')

