#!/usr/bin/env python
# coding: utf-8

"""
BIBLIOTECA DE CÓDIGOS
Regressão Linear - Parte II
Ana Cristina Silva - OpusNove

Código em Python 3.6.8
Abril 2021
"""

# Importações

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Conhecendo o conjunto de dados

df = pd.read_csv('dataset - Dados imobiliários.csv', sep = ';')
df.head()

print('Tamanho do conjunto de dados: ')
print(df.shape)
print('-------------')
print('Informação das variáveis: ')
df.info()

df.describe().round(2)

df.corr().round(4)


# Avaliação do Comportamento da Variável Y, Dependente

ax = sns.boxplot(data=df['Valor'], orient='h', width=0.3)
ax.figure.set_size_inches(20, 5)
ax.set_title('Preço dos Imóveis', fontsize=20)
ax.set_xlabel('Reais', fontsize=16)
ax

ax = sns.distplot(df['Valor'])
ax.figure.set_size_inches(20, 6)
ax.set_title('Distribuição de Frequências', fontsize=20)
ax.set_xlabel('Preço dos Imóveis (R$)', fontsize=16)
ax

ax = sns.pairplot(df, y_vars='Valor', x_vars=['Area', 'Dist_Praia', 'Dist_Farmacia'], kind='reg', height=5)
ax.fig.suptitle('Dispersão entre as Variáveis', fontsize=20, y=1.05)

# Será necessário transformar os dados com a finalidade de torná-lo mais proximos da distribuição normal. 

df['log_Valor'] = np.log(df['Valor'])
df['log_Area'] = np.log(df['Area'])
df['log_Dist_Praia'] = np.log(df['Dist_Praia'] + 1)
df['log_Dist_Farmacia'] = np.log(df['Dist_Farmacia'] + 1)
df.head()

ax = sns.distplot(df['log_Valor'])
ax.figure.set_size_inches(12, 6)
ax.set_title('Distribuição de Frequências', fontsize=20)
ax.set_xlabel('log do Preço dos Imóveis', fontsize=16)
ax

ax = sns.pairplot(df, y_vars='log_Valor', x_vars=['log_Area', 'log_Dist_Praia', 'log_Dist_Farmacia'], height=5)
ax.fig.suptitle('Dispersão entre as Variáveis Transformadas', fontsize=20, y=1.05)
ax


# Regressão Linear

# Criação do Modelo - Primeiro Modelo

from sklearn.model_selection import train_test_split

y = df['log_Valor']
X = df[['log_Area', 'log_Dist_Praia', 'log_Dist_Farmacia']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2811)

from sklearn.linear_model import LinearRegression
from sklearn import metrics

modelo = LinearRegression()

modelo.fit(X_train, y_train)


# Avaliação do Modelo - Primeiro Modelo

import statsmodels.api as sm

X_train_com_constante = sm.add_constant(X_train)
modelo_statsmodels = sm.OLS(y_train, X_train_com_constante, hasconst = True).fit()
print(modelo_statsmodels.summary())

# Primeiras conclusões
# * A feature log_Dist_Farmacia não contribui para o modelo e pode ser retirada
# * A estatística F apresentou resultado desejado
# * O valor de R2 apresenta resultado adquado

# Cálculo de R² separadamente

print('R² = {}'.format(modelo.score(X_train, y_train).round(3)*100))

# Os dados são explicados pelas variáveis independentes em 80,5%

y_pred = modelo.predict(X_train)
y_pred2 = y_pred[:1000]

print('RMSE = {}'.format(metrics.mean_squared_error(y_test, y_pred2).round(3))) # Muito utilizado para daods contínuos
print('MAE = {}'.format(metrics.mean_absolute_error(y_test, y_pred2).round(3)))

# Quanto menor os erros melhor é o resultado do modelo.

X = df[['log_Area', 'log_Dist_Praia']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2811)
X_train_com_constante = sm.add_constant(X_train)
modelo_statsmodels = sm.OLS(y_train, X_train_com_constante, hasconst = True).fit()

modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Avaliação do Modelo - Segundo Modelo

print(modelo_statsmodels.summary())


#  Primeiras conclusões
# * A feature log_Dist_Farmacia foi retirada do modelo e não afetou seu desempenho 
# * A estatística F apresentou resultado desejado
# * O valor de R2 apresenta resultado adquado e continua  o mesmo

# Cálculo de R² separadamente

print('R² = {}'.format(modelo.score(X_train, y_train).round(3)*100))

# **O valor de R² permanece o mesmo**

y_pred = modelo.predict(X_train)
y_pred2 = y_pred[:1000]

print('RMSE = {}'.format(metrics.mean_squared_error(y_test, y_pred2).round(3))) # Muito utilizado para daods contínuos
print('MAE = {}'.format(metrics.mean_absolute_error(y_test, y_pred2).round(3)))


# Comparando os dois modelos
# Modelo|R²|RMSE|MAE|Estatística F
# :---:|:---:|:---:|:---:|:----:
# Modelo 1|80.5|1.433|1.432|00
# Modelo 2|80.5|0.961|0.961|00

# Escolhemos o Modelo 2 porque apresenta melhor erro MAE e porque possui praticamente o mesmo resultado 
# do modelo 1 com menos variáveis explicativas**

# Previsões pontuais - Modelo 2

# Dados de entrada
entrada = X_test[0:1]

# Previsão
modelo.predict(entrada)[0]

# Conversão do resultado
print('R$ {0:.2f}'.format(np.exp(modelo.predict(entrada)[0])))


# Interpretação do Coeficientes Estimados - Modelo 2

index = ['Intercepto', 'log Área', 'log Distância até a Praia']
pd.DataFrame(data=np.append(modelo.intercept_, modelo.coef_), index=index, columns=['Parâmetros'])

y_previsto_train = modelo.predict(X_train)

ax = sns.scatterplot(x=y_previsto_train, y=y_train)
ax.figure.set_size_inches(12, 6)
ax.set_title('Previsão X Real', fontsize=18)
ax.set_xlabel('log do Preço - Previsão', fontsize=14)
ax.set_ylabel('log do Preço - Real', fontsize=14)
ax

# Resíduos
residuo = y_train - y_previsto_train

ax = sns.distplot(residuo)
ax.figure.set_size_inches(12, 6)
ax.set_title('Distribuição de Frequências dos Resíduos', fontsize=18)
ax.set_xlabel('log do Preço', fontsize=14)
ax
