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

df['Species'].unique()

# Conhecendo os dados

print('Tamanho do conjunto de dados: ')
print(df.shape)
print('-------------')
print('Informação das variáveis: ')
df.info()

# Distribuição de Frequências

# Usando o crosstab
# São tabelas que exibem o relacionamento entre duas ou mais variáveis

frequencia = pd.crosstab(df.Species,
                         df.PetalLengthCm, 
                         aggfunc = 'mean',
                         values = df.PetalWidthCm)
frequencia


# Interpretação da tabela - Exemplo: 
# Para a espécie **Iris-setosa** com comprimento da pétala em 1.0 centímetros, 
# a largura média da pétala é de 0.2 centrímetros. Análise da célula 1,1 da tabela. 

# Distribuição de frequência por classes

# Foi utilizado o tamanho da pétala

tamanho_minimo = df.PetalLengthCm.min()
tamanho_maximo = df.PetalLengthCm.max()

print('Tamanho mínimo da pétala: ', tamanho_minimo)
print('Tamanho máximo da pétala: ', tamanho_maximo)

# DEFINIÇÃO DAS CLASSES: 
#  Utilizar a seguinte classificação:
#  <b>A</b> ► Acima de 6
#  <b>B</b> ► De 5 a 5.9
#  <b>C</b> ► De 4 a 4.9
#  <b>D</b> ► De 3 a 3.9
#  <b>E</b> ► Até 2 

# Definindo a tabela de frequencia de tamanhos

classes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.9]
labels = ['E', 'D', 'C', 'B', 'A']

frequencia = pd.value_counts(
  pd.cut(x = df.PetalLengthCm,
         bins = classes,
         labels = labels,
         include_lowest = True)
)
frequencia

percentual = pd.value_counts(
  pd.cut(x = df.PetalLengthCm,
         bins = classes,
         labels = labels,
         include_lowest = True),
  normalize = True
) * 100
percentual.round(2)


# Distribuição de frequência por classes fixas

# Definindo o número de classes

n = df.shape[0]
k = 1 + (10 /3) * np.log10(n)
k = int(k.round(0))

print('Total de classes definidas: ', k)

frequencia = pd.value_counts(
  pd.cut(
    x = df.PetalLengthCm,
    bins = 17,
    include_lowest = True
  ),
  sort = False
)

percentual = (pd.value_counts(
  pd.cut(
    x =  df.PetalLengthCm,
    bins = 17,
    include_lowest = True
  ),
  sort = False,
  normalize = True
) * 100).round(2)

dist_freq_quantitativas_amplitude_fixa = pd.DataFrame(
    {'Frequência': frequencia, 'Porcentagem (%)': percentual}
)
dist_freq_quantitativas_amplitude_fixa


#  Histograma
# O **HISTOGRAMA** é a representação gráfica de uma distribuição de frequências. 
# É um gráfico formado por um conjunto de retângulos colocados lado a lado, 
# onde a área de cada retângulo é proporcional à frequência da classe que ele representa.

ax = sns.distplot(df.PetalLengthCm)

ax.figure.set_size_inches(12, 6)
ax.set_title('Distribuição de Frequências -Tamanho das pétalasE', fontsize=18)
ax.set_xlabel('Centímetros', fontsize=14)
ax

dist_freq_quantitativas_amplitude_fixa['Frequência'].plot.bar(width= 1, color = 'red', alpha = 0.5, figsize=(12, 6))


# Distribuição de frequência acumulada

ax = sns.distplot(df.SepalLengthCm, 
                  hist_kws = {'cumulative': True},
                  kde_kws = {'cumulative': True},
                  bins = 10)
ax.figure.set_size_inches(14, 6)
ax.set_title('Distribuição de Frequências Acumulada', fontsize=18)
ax.set_ylabel('Acumulado', fontsize=14)
ax.set_xlabel('Tamanho', fontsize=14)
ax
