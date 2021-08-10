#!/usr/bin/env python
# coding: utf-8

"""
BIBLIOTECA DE CÓDIGOS

Distribuição de Probabilidade - Amostragem
Ana Cristina Silva - OpusNove

Código em Python 3.6.8
Abril 2021

Fonte de dados: https://www.kaggle.com/uciml/iris
"""

# Importações

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Abrindo o conjunto de dados

df = pd.read_csv('Iris.csv', encoding='utf-8')
df.head()

# Conhecendo os dados

print('Tamanho do conjunto de dados: ')
print(df.shape)
print('-------------')
print('Informação das variáveis: ')
df.info()

"""
População
    Conjunto de todos os elementos de interesse em um estudo. 
    Diversos elementos podem compor uma população, por exemplo: pessoas, idades, alturas, carros etc. 
    Com relação ao tamanho, as populações podem ser limitadas (populações finitas) 
    ou ilimitadas (populações infinitas).

Populações finitas
    Permitem a contagem de seus elementos. 
    Como exemplos temos o número de funcionário de uma empresa, 
    a quantidade de alunos em uma escola etc.
 
Populações infinitas
    Não é possível contar seus elementos. 
    Como exemplos temos a quantidade de porções que se pode extrair 
    da água do mar para uma análise, temperatura medida em cada ponto de um território etc. 
    O estudo não chegaria nunca ao fim. Não é possível investigar todos os elementos da população.

Quando os elementos de uma população puderem ser contados, porém apresentando 
uma quantidade muito grande, assume-se a população como infinita.
 
Amostra
    Subconjunto representativo da população. 
    Os atributos numéricos de uma população como sua média, variância e desvio padrão, 
    são conhecidos como **parâmetros**. O principal foco da inferência estatística é 
    justamente gerar estimativas e testar hipóteses sobre os parâmetros populacionais utilizando 
    as informações de amostras.

Testes destrutivos
    Estudos onde os elementos avaliados são totalmente consumidos ou destruídos. 
    Exemplo: testes de vida útil, testes de segurança contra colisões em automóveis.

Resultados rápidos
    Pesquisas que precisam de mais agilidade na divulgação. Exemplo: pesquisas de opinião, 
    pesquisas que envolvam problemas de saúde pública.
 
Custos elevados
Quando a população é finita mas muito numerosa, o custo de um censo pode tornar o processo inviável.

Amostragem Aleatória Simples
    É uma das principais maneiras de se extrair uma amostra de uma população. 
    A exigência fundamental deste tipo de abordagem é que cada elemeto da população tenha as mesmas 
    chances de ser selecionado para fazer parte da amostra.

Amostragem Estratificada
    É uma melhoria do processo de amostragem aleatória simples. 
    Neste método é proposta a divisão da população em subgrupos de elementos com características similares, 
    ou seja, grupos mais homogêneos. Com estes subgrupos separados, aplica-se a técnica de amostragem 
    aleatória simples dentro de cada subgrupo individualmente.

Amostragem por Conglomerado
    Também visa melhorar o critério de amostragem aleatória simples. 
    Na amostragem por conglomerados são também criados subgrupos, porém não serão homogêneas 
    como na amostragem estratificada. Na amostragem por conglomerados os subgrupos serão 
    heterogêneos, onde, em seguida, serão aplicadas a amostragem aleatória simples ou estratificada.
    Um exemplo bastante comum de aplicação deste tipo de técnica é na divisão da população em grupos territoriais,
    onde os elementos investigados terão características bastante variadas.

ESTIMAÇÃO
    É a forma de se fazer suposições generalizadas sobre os parâmetros de uma população 
    tendo como base as informações de uma amostra.
      - Parâmetros são os atributos numéricos de uma população, tal como a média, desvio padrão etc.
      - Estimativa é o valor obtido para determinado parâmetro a partir dos dados de uma amostra da população.

Teorema do Limite Central
    O Teorema do Limite Central afirma que, com o aumento do tamanho da amostra, 
    a distribuição das médias amostrais se aproxima de uma distribuição normal com média 
    igual à média da população e desvio padrão igual ao desvio padrão da variável 
    original dividido pela raiz quadrada do tamanho da amostra. 
    Este fato é assegurado para $n$ maior ou igual a 30.

O desvio padrão das médias amostrais é conhecido como erro padrão da média

Níveis de confiança e significância
    * O nível de confiança representa a probabilidade de acerto da estimativa. 
    * De forma complementar o nível de significância expressa a probabilidade de erro da estimativa.
    * O nível de confiança representa o grau de confiabilidade do resultado da estimativa estar 
    dentro de determinado intervalo. Quando fixamos em uma pesquisa um nível de confiança de 95%, 
    por exemplo, estamos assumindo que existe uma probabilidade de 95% dos resultados da pesquisa representarem bem a realidade, 
    ou seja, estarem corretos. 
    * O nível de confiança de uma estimativa pode ser obtido a partir da área sob a curva normal como ilustrado na figura abaixo.

"""

# Conjunto inteiro dos dados

print('Tamanho do conunto de dados', df.shape[0])
print('Média do tamanho das pétalas: ', df.PetalWidthCm.mean())

# Cálculo para amostra simples

amostra = df.sample(n = 105, random_state = 101) # Amostra de 70% dos elementos

print('Tamanho do conunto de dados', amostra.shape[0])
print('Média do tamanho das pétalas: ', amostra.PetalWidthCm.mean())

# Estamos estudando o rendimento mensal dos chefes de domicílios no Brasil. 
# Nosso supervisor determinou que o erro máximo em relação a média seja de R$\$$ 100,00. 
# Sabemos que o desvio padrão populacional deste grupo de trabalhadores é de R$\$$ 3.323,39. 
# Para um nível de confiança de 95%, qual deve ser o tamanho da amostra de nosso estudo?

z = norm.ppf(0.5 + (0.95 / 2))
sigma = 3323.39
e = 100
n = int(((z * (sigma / e)) ** 2).round())

print('A variável normal padronizada é: ', z)
print('O desvio padrão populacional é : ', sigma)
print('O erro inferencial é: ', e)
print('O tamanho da amostra é: ', n)

# Em um lote de 10.000 latas de refrigerante foi realizada uma amostra aleatória simples de 100 latas 
# e foi obtido o desvio padrão amostral do conteúdo das latas igual a 12 ml. 
# O fabricante estipula um erro máximo sobre a média populacional de apenas 5 ml. 
#Para garantir um nível de confiança de 95% qual o tamanho de amostra deve ser selecionado para este estudo?

N = 10000
z = norm.ppf((0.5 + (0.95 / 2)))
s = 12
e = 5
n = int((((z**2) * (s**2) * (N)) / (((z**2) * (s**2)) + ((e**2) * (N - 1)))).round())

print('Tamanho da população: ', N)
print('Variável normal padronizada: ', z)
print('Desvio padrão amostral: ', s)
print('Erro inferencial: ', e)
print('O tamanho da amostra é: ', n)

# Criando o conjunto de amostras

n = 105
total_de_amostras = 80

amostras = pd.DataFrame()

for i in range(total_de_amostras):
  _ = df.PetalWidthCm.sample(n)
  _.index = range(0, len(_))
  amostras['Amostra_' + str(i)] = _

amostras

# Verificando a média das amostras

amostras.mean()

# Histogama das médias das amostras

amostras.mean().hist()

print('Média do conjunto de dados avaliado: ', df.PetalWidthCm.mean())
print('Média das médias das amostras: ', amostras.mean().mean())
print('Diferença do cálculo acima: ',  df.PetalWidthCm.mean() - amostras.mean().mean())

# Suponha que os pesos dos sacos de arroz de uma indústria alimentícia se distribuem aproximadamente 
# como uma normal de desvio padrão populacional igual a 150 g. 
# Selecionada uma amostra aleatório de 20 sacos de um lote específico, obteve-se um peso médio de 5.050 g. 
# Construa um intervalo de confiança para a média populacional assumindo um nível de significância de 5%.

from scipy.stats import norm

media_amostra = 5050
significancia = 0.05
confianca = 1 - significancia
z = norm.ppf(0.975)

desvio_padrao = 150
n = 20
raiz_de_n = np.sqrt(n)
sigma = desvio_padrao / raiz_de_n

e = z * sigma

print('Média da amostra: ', media_amostra)
print('Significância da amostra: ', significancia)
print('Nível de confiança da amostra: ', confianca)
print('A estatística de teste é: ', z)
print('O erro padrão da média é: ', sigma)
print('O erro inferencial é: ', e)
print('O intervalo de confiança é: ', norm.interval(alpha = 0.95, loc = media_amostra, scale = sigma))
