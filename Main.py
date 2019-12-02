import os
import sys
import random
import numpy as np
import time
import sklearn.decomposition as decomp
from PIL import Image
from scipy import sum, average

# Permite a colorização do console
from sty import fg, bg, ef, rs, Style, RgbFg
if sys.platform == "win32":
    os.system('color')

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
        if img.endswith(".jpg") or img.endswith(".png"):
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


def formata_porcentagem(pcent, gr):
    if pcent < 50:
        return f"{gr.red}{pcent:10f}"
    elif pcent >= 50 and pcent < 80:
        return f"{gr.yellow}{pcent:10f}"
    elif pcent >= 80:
        return f"{gr.green}{pcent:10f}"


def calcular_taxa_acertos(entrada, saida):
    acertos = 0
    for i in range(len(entrada)):
        if entrada[i] == saida[i]:
            acertos += 1

    return acertos*100/len(saida)


def calcula_mean_vector(matriz):
    matriz = np.array(matriz)

    return matriz.mean(axis=0)


def subtrai_mean_vector(matriz):

    mean_vector = calcula_mean_vector(matriz)

    for coluna in matriz:
        for item in range(len(matriz)):
            coluna[item] = abs(coluna[item] - mean_vector[item])

    print(matriz)
    return matriz


def montar_data_matrix(dataset, nome_pasta, extensao):
    data_matrix = []

    for img_name in dataset:
        imagem = np.asarray(
            Image.open("./" + nome_pasta + "/" + img_name).convert('LA'), dtype="float32")
        imagem = average(imagem, -1)
        coluna_imagem = np.empty((len(imagem) * len(imagem[0]), 1))

        for i in imagem:
            coluna_imagem = coluna_imagem + i

        data_matrix.append(coluna_imagem)

    return data_matrix


# Realiza testes buscando a menor diferença entre as normas
def reconhecimento_norma(nome_imagem, treino_set, nome_pasta, extensao):
    imagem = np.asarray(
        Image.open("./" + nome_pasta + "/" + nome_imagem + extensao).convert('LA'), dtype="float32")
    imagem = average(imagem, -1)

    menor_diferenca = comparative
    output = ''
    diferencas = {}

    for img in treino_set:
        img_treino = np.asarray(
            Image.open("./" + nome_pasta + "/" + img + extensao).convert('LA'), dtype="float32")
        img_treino = average(img_treino, -1)

        norma = compare_images(imagem, img_treino)
        diferencas.update({img.split('-')[0]: norma})
        if norma < menor_diferenca:
            menor_diferenca = norma
            output = img.split('-')[0]

    diferencas = normaliza_array(diferencas)
    certeza = 100 - diferencas[output]
    return certeza, output, diferencas


# Aplica a decomposição baseado no método PCA() da lib
def aplica_pca(imagem):
    pca = decomp.PCA(n_components=2)
    pca.fit(imagem)
    return pca.transform(imagem)


# Realiza os testes aplicando o método PCA
def reconhecimento_pca(nome_imagem, treino_set, nome_pasta, extensao):
    imagem = np.asarray(
        Image.open("./" + nome_pasta + "/" + nome_imagem + extensao).convert('LA'), dtype="float32")
    imagem = average(imagem, -1)

    menor_diferenca = comparative
    output = ''
    diferencas = {}

    for img in treino_set:
        img_treino = np.asarray(
            Image.open("./" + nome_pasta + "/" + img + extensao).convert('LA'), dtype="float32")
        img_treino = average(img_treino, -1)

        norma = compare_images(aplica_pca(imagem), aplica_pca(img_treino))
        diferencas.update({img.split('-')[0]: norma})
        if norma < menor_diferenca:
            menor_diferenca = norma
            output = img.split('-')[0]

    diferencas = normaliza_array(diferencas)
    certeza = 100 - diferencas[output]
    return certeza, output, diferencas


# Realiza o teste de reconhecimento
def reconhece(nome_pasta, extensao, num_imagens_teste):
    inicio = time.time()
    treino, teste = separa_conjuntos(num_imagens_teste, nome_pasta)
    img_teste = []
    img_output = []

    for img in teste:
        certeza, output, diferencas = reconhecimento_norma(img, treino, nome_pasta, extensao)
        
        img = img.split('-')[0]
        
        img_teste.append(img)
        img_output.append(output)

        print(
            f"-> RECONHECENDO ("
            f"{fg.magenta}ID: {int(img):02}{rs.fg}, "
            f"{fg.yellow}Foto: {int(img):02}{rs.fg})"
        )

        print(
            f"\t |-> Melhor semelhança = "
            f"{fg.magenta}{ef.u}ID: {int(output):02}{rs.u}{rs.fg}, "
            f"Certeza = "
            f"{ef.u}{formata_porcentagem(certeza, fg)}%{rs.u}{rs.fg} ")

    taxa_acertos = calcular_taxa_acertos(img_teste, img_output)
    fim = time.time()

    return taxa_acertos, fim - inicio


def main():
    taxa_acertos, tempo = reconhece('medium', '.jpg', 3)
    print(f"\n{'======== ESTATÍSTICAS ========':^50}")
    print(f"\nO processamento levou {bg.da_blue}{tempo} segundos{rs.bg}")
    print(f"A taxa de acertos foi de {fg.black}{formata_porcentagem(taxa_acertos, bg)}%{rs.all}")


# main()

data = montar_data_matrix(os.listdir("./" + 'easy'), 'easy', '')

subtrai_mean_vector(data)
