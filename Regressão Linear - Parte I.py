#!/usr/bin/env python
# coding: utf-8

"""
BIBLIOTECA DE CÓDIGOS
Regressão Linear - Parte I
Ana Cristina Silva - OpusNove

Código em Python 3.6.8
Abril 2021

Conjunto de dados utilizados: 
Fonte: https://www.kaggle.com/dongeorge/beer-consumption-sao-paulo
"""

# Importações

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Conhecendo o conjunto de dados

df = pd.read_csv('Consumo_cerveja.csv', sep = ';')
df.head()

print('Tamanho do conjunto de dados: ')
print(df.shape)
print('-------------')
print('Informação das variáveis: ')
df.info()

df.describe().round(2)


# Matriz de correlação

df.corr().round(4)

# Avaliação do Comportamento da Variável Y, Dependente

ax = sns.boxplot(y = 'consumo', x = 'fds', data = df, orient = 'v', width = 0.5)
ax.figure.set_size_inches(12,6)
ax.set_title('Consumo de Cerveja', fontsize = 35)
ax.set_ylabel('Litros', fontsize = 20)
ax.set_xlabel('Final de semana', fontsize = 20)
ax

ax = sns.distplot(df['consumo'])
ax.figure.set_size_inches(12,6)
ax.set_title('Distribuição de frequências', fontsize = 35)
ax.set_ylabel('Consumo de cerveja (Litros)', fontsize = 20)

ax = sns.pairplot(df, y_vars = 'consumo', x_vars = ['temp_min', 'temp_media', 'temp_max' , 'chuva' , 'fds'])
ax.fig.suptitle('Dispersão entre as variáveis', fontsize = 20 , y = 1.2)
ax

# Avaliação do Comportamento das Variáveis Explicativas

ax = sns.pairplot(df)

# Análise entre a variável dependente e a variável explicativa Temperatura

ax = sns.jointplot(x = 'temp_max' , y = 'consumo' , data = df , kind = 'reg')
ax.fig.suptitle('Dispersão Consumo x Temperatura', fontsize = 20 , y = 1.05)
ax.set_axis_labels('Temperatura Máxima' , 'Consumo de Cerveja', fontsize = 14)
ax

# Visualização da Regressão Linear

ax = sns.lmplot(x = 'temp_max' , y = 'consumo' , data = df)
ax.fig.suptitle('Reta de regressão - Consumo x Temperatura', fontsize = 20 , y = 1.05)
ax.set_xlabels('Temperatura Máxima - ºC', fontsize = 14)
ax.set_ylabels('Consumo de cerveja - l', fontsize = 14)
ax

ax = sns.lmplot(x = 'temp_max' , y = 'consumo' , data = df , hue = 'fds' , markers = ['o' , 'x'] , legend = False)
ax.fig.suptitle('Reta de regressão - Consumo x Temperatura x Final de semana', fontsize = 20 , y = 1.05)
ax.set_xlabels('Temperatura Máxima - ºC', fontsize = 14)
ax.set_ylabels('Consumo de cerveja - l', fontsize = 14)
ax.add_legend(title = 'Fim de Semana')
ax

ax = sns.lmplot(x = 'temp_max' , y = 'consumo' , data = df , col = 'fds')
ax.fig.suptitle('Reta de regressão - Consumo x Temperatura x Final de semana', fontsize = 20 , y = 1.05)
ax.set_xlabels('Temperatura Máxima - ºC', fontsize = 14)
ax.set_ylabels('Consumo de cerveja - l', fontsize = 14)
ax


# Regressão Linear
# Biblioteca para a fromatação de dados

from sklearn.model_selection import train_test_split

# Variável dependente

y = df['consumo']
y.head()

# Variáveis explicativas

X = df[['temp_max' , 'chuva' , 'fds']]
X.head()

# Grupo de treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2811)

# Bibliotecas para a regressão linear

from sklearn.linear_model import LinearRegression 
from sklearn import metrics

# Criação do modelo

modelo = LinearRegression()
modelo.fit(X_train, y_train)
modelo

# Verificação do ajuste da regressão

print('R² = {}'.format(modelo.score(X_train, y_train).round(2))) # Para fazer o dois pequenininho aperte Alt + 1021

# Gerando previsões

y_previsto = modelo.predict(X_test)
print('As cinco primeiras previsões são: ', y_previsto[:5])

# Verificando o ajuste das previsões

print('R² = %s' %metrics.r2_score(y_test, y_previsto).round(2))

# Forma do modelo

index = ['Intercepto' , 'Temperatura Máxima (ºC)' , 'Chuva (mm)', 'Final de semana']
pd.DataFrame(data = np.append(modelo.intercept_, modelo.coef_), index = index, columns = ['Parâmetro'])

y_previsto_treino = modelo.predict(X_train)

ax = sns.scatterplot(x = y_previsto_treino, y = y_train)
ax.figure.set_size_inches(12,6)
ax.set_title('Previsão x Real', fontsize = 18)
ax.set_xlabel('Consumo de Cerveja (l) - Previsão', fontsize = 14)
ax.set_ylabel('Consumo de Cerveja (l) - Real', fontsize = 14)
ax

residuo = y_train - y_previsto_treino

ax = sns.scatterplot(x = y_previsto_treino, y = residuo, s = 150)
ax.figure.set_size_inches(20,8)
ax.set_title('Residuo x Previsão', fontsize = 18)
ax.set_xlabel('Consumo de Cerveja (l) - Previsão', fontsize = 14)
ax.set_ylabel('Resíduo', fontsize = 14)
ax

ax = sns.distplot(residuo, bins = 50) # Bins aumenta o número de barras
ax.figure.set_size_inches(12,6)
ax.set_title('Distribuição de frequencia dis Resíduos', fontsize = 18)
ax.set_xlabel('Litros', fontsize = 14)
ax

"""
Métricas para a Regressão Linear

    Erro Quadrático Médio 
        Média dos quadrados dos erros. Ajustes melhores apresentam $EQM$ mais baixo.

    Raíz do Erro Quadrático Médio
        Raíz quadrada da média dos quadrados dos erros. Ajustes melhores apresentam $\sqrt{EQM}$ mais baixo.
"""

# Quanto menor o erro que for encontrado , melhor.

EQM_2 = metrics.mean_squared_error(y_test, y_previsto).round(2)
REQM_2 = np.sqrt(metrics.mean_squared_error(y_test, y_previsto)).round(2)
R2_2 = metrics.r2_score(y_test, y_previsto).round(2)

pd.DataFrame([EQM_2, REQM_2, R2_2] , ['EQM' , 'REQM' , 'R²'] , columns = ['Métricas'])
