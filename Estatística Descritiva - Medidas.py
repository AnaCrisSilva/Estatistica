#!/usr/bin/env python
# coding: utf-8

"""
BIBLIOTECA DE CÓDIGOS

Estatística Descritiva
Ana Cristina Silva - OpusNove

Código em Python 3.6.8
Abril 2021

Fonte de dados: https://www.kaggle.com/uciml/iris
"""
# Importações

import pandas as pd
import numpy as np
import seaborn as sns

# Abrindo o conjunto de dados

df = pd.read_csv('Iris.csv', encoding='utf-8')
df.head()

# Conhecendo os dados

print('Tamanho do conjunto de dados: ')
print(df.shape)
print('-------------')
print('Informação das variáveis: ')
df.info()

# Medidas de Tendência Central

# Média aritmética

print('A média do tamanho das pétalas é: ', df['PetalLengthCm'].mean())
print('A média da largura das pétalas é: ', df['PetalWidthCm'].mean())

# Mediana
 
# Para obtermos a mediana de uma conjunto de dados devemos proceder da seguinte maneira:
# 1. Ordenar o conjunto de dados;
# 2. Identificar o número de observações (registros) do conjunto de dados ($n$);
# 3. Identicar o elemento mediano, quando par e quando ímpar.

print('A mediana do tamanho das pétalas é: ', df['PetalLengthCm'].median())
print('A mediana da largura das pétalas é: ', df['PetalWidthCm'].median())


#  Moda

# Pode-se definir a moda como sendo o valor mais frequente de um conjunto de dados. 
# A moda é bastante utilizada para dados qualitativos.

print('A moda do tamanho das pétalas é: ', df['PetalLengthCm'].mode())
print('A moda da largura das pétalas é: ', df['PetalWidthCm'].mode())

# Medidas Separatrizes

# Quartis, decis e percentis
# Os quartis permitem dividir a distribuição em quatro partes iguais quanto 
# ao número de elementos de cada uma; os decis em dez partes e os centis em cem partes iguais.

# Quartiles mais utilizados

df.SepalLengthCm.quantile([0.25, 0.5, 0.75])
df.SepalWidthCm.quantile([0.25, 0.5, 0.75])

# Dez quartiles

df.SepalLengthCm.quantile([i / 10 for i in range(1, 10)])


# Box Plot
# Representação gráfica das principais medidas separatrizes

ax = sns.boxplot(x = 'PetalLengthCm', data = df, orient = 'h')
ax.figure.set_size_inches(12, 4)
ax.set_title('Tamanho da Pétala', fontsize=18)
ax.set_xlabel('Centímetros', fontsize=14)
ax

ax = sns.boxplot(x = 'PetalLengthCm', y = 'Species', data = df, orient = 'h')
ax.figure.set_size_inches(12, 4)
ax.set_title('Tamanho da Pétala', fontsize=18)
ax.set_xlabel('Centímetros', fontsize=14)
ax

# Medidas de Dispersão

# Desvio médio absoluto

df_SepalLengthCm = df[['SepalLengthCm']][:6]
df_SepalLengthCm

tamanho_medio = df_SepalLengthCm.mean()[0]
tamanho_medio

# Desvios absolutos

df_SepalLengthCm['SepalLengthCm'].abs()

ax = df_SepalLengthCm['SepalLengthCm'].plot(style = 'o')
ax.figure.set_size_inches(14, 6)
ax.hlines(y = tamanho_medio, xmin = 0, xmax = df_SepalLengthCm.shape[0] - 1, colors='red')
for i in range(df_SepalLengthCm.shape[0]):
    ax.vlines(x = i, ymin = tamanho_medio, ymax = df_SepalLengthCm['SepalLengthCm'][i], linestyles='dashed')
ax

desvio_medio_absoluto = df_SepalLengthCm['SepalLengthCm'].mad()
desvio_medio_absoluto


# Variância
 
# A variância é construída a partir das diferenças entre cada observação e a média dos dados, 
# ou seja, o desvio em torno da média. No cálculo da variância, os desvios em torno da média são elevados ao quadrado.

# Variância para a largura da sépala

df['SepalWidthCm'].var()


# Desvio Padrão
# Uma das restrições da variância é o fato de fornecer medidas em quadrados das unidades originais - 
# a variância de medidas de comprimento, por exemplo, é em unidades de área.
#  Logo, o fato de as unidades serem diferentes dificulta a comparação da dispersão com as variáveis que a definem.
#  Um modo de eliminar essa dificuldade é considerar sua raiz quadrada.

df['SepalWidthCm'].std()
