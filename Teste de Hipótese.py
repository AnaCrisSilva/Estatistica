#!/usr/bin/env python
# coding: utf-8

"""
BIBLIOTECA DE CÓDIGOS
Teste de Hipótese
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

"""
 Passos para o TESTE
 
 Passo 1 - formulação das hipóteses H_0 e H_1
 
 Pontos importantes
  - De maneira geral, o alvo do estudo deve ser formulado como a hipótese alternativa H_1.
  - A hipótese nula sempre afirma uma igualdade ou propriedade populacional, 
    e H_1 a desigualdade que nega H_0
  - No caso da hipótese nula H_0 a igualdade pode ser representada por uma igualdade 
    simples "=" ou por "\geq" e "\leq". Sempre complementar ao estabelecido pela hipótese alternativa.
  - A hipótese alternativa H_1 deve definir uma desigualdade que pode ser 
    uma diferença simples "\neq" ou dos tipos ">" e "<".
 
 ------------------------------------------------------------------------------------
 
 Passo 2 - escolha da distribuição amostral adequada;
 
 Pontos importantes
  - Quando o tamanho da amostra tiver 30 elementos ou mais, 
    deve-se utilizar a distribuição normal, como estabelecido pelo teorema do limite central.
  - Para um tamanho de amostra menor que 30 elementos, 
    e se pudermos afirmar que a população se distribui aproximadamente como uma normal 
    e o desvio padrão populacional for conhecido, deve-se utilizar a distribuição normal.
  - Para um tamanho de amostra menor que 30 elementos, 
    e se pudermos afirmar que a população se distribui aproximadamente como uma normal 
    e o desvio padrão populacional for desconhecido, deve-se utilizar a distribuição t de Student.
 
 ------------------------------------------------------------------------------------
 
 Passo 3 - fixação da significância do teste (\alpha), 
           que define as regiões de aceitação e rejeição das hipóteses 
           (os valores mais freqüentes são 10%, 5% e 1%);
 
 Pontos importantes
  - O nível de confiança (1 - \alpha) representa a probabilidade de acerto da estimativa. 
    De forma complementar o nível de significância (\alpha) expressa a probabilidade de erro da estimativa.
  - O nível de confiança representa o grau de confiabilidade do resultado da 
    estimativa estar dentro de determinado intervalo. Quando fixamos em uma pesquisa um 
    nível de confiança de 95%, por exemplo, estamos assumindo que existe uma 
    probabilidade de 95% dos resultados da pesquisa representarem bem a realidade, ou seja, estarem corretos.
 
  ------------------------------------------------------------------------------------

 Passo 4 - cálculo da estatística-teste e verificação desse valor com as áreas de aceitação e rejeição do teste;
 
 Pontos importantes
  - Nos testes paramétricos, distância relativa entre a estatística amostral e o valor alegado como provável.
  - Neste passo são obtidas as estatísticas amostrais necessárias à execução do 
    teste (média, desvio-padrão, graus de liberdade etc.)
 
 ------------------------------------------------------------------------------------
 
 Passo 5 - Aceitação ou rejeição da hipótese nula.
 
 Pontos importantes
  - No caso de o intervalo de aceitação conter a estatística-teste, aceita-se H_0 
    como estatisticamente válido e rejeita-se $H_1$ como tal.
  - No caso de o intervalo de aceitação não conter a estatística-teste, 
    rejeita-se H_0 e aceita-se H_1 como provavelmente verdadeira. 
  - A aceitação também se verifica com a probabilidade de cauda (p-valor): 
    se maior que \alpha, aceita-se H_0.

"""

# Teste de Normalidade
# A função *normaltest* testa a hipótese nula H_0 de que a amostra é proveniente de uma distribuição normal.

from scipy.stats import normaltest

# Critério do valor p
# Rejeitar H_0 se o valor p\leq 0,05

df.SepalWidthCm.hist(bins = 50)


# - Estatística de teste: Grau de concordância entre uma amostra e a hipótese nula.
# - P-valor: Probabilidade de de que a estatística do teste (como variável aleatória) 
# tenha valor extremo em relação ao valor observado (estatística) quando a hipótese H0 é verdadeira.

stat_test, p_valor = normaltest(df.SepalWidthCm)
print('A estatística de teste é: ', stat_test)
print('O p-valor é: ', p_valor)

significancia = 0.05

p_valor <= significancia


# A hipótese nula é aceita. Isso significa que a distribuição provavelmente é normal.

# Testes Paramétricos

"""
Teste Bicaudal

A empresa Suco Bom produz sucos de frutas em embalagens de 500 ml. 
Seu processo de produção é quase todo automatizado e as embalagens de sucos são 
preenchidas por uma máquina que às vezes apresenta um certo desajuste, 
levando a erros no preenchimento das embalagens para mais ou menos conteúdo. 
Quando o volume médio cai abaixo de 500 ml, a empresa se preocupa em perder 
vendas e ter problemas com os orgãos fiscalizadores. Quando o volume passa de 500 ml, 
a empresa começa a se preocupar com prejuízos no processo de produção.
 
O setor de controle de qualidade da empresa Suco Bom extrai, periodicamente, 
amostras de 50 embalagens para monitorar o processo de produção. 
Para cada amostra, é realizado um teste de hipóteses para avaliar se o maquinário se desajustou. 
A equipe de controle de qualidade assume um nível de significância de 5%.

Suponha agora que uma amostra de 50 embalagens foi selecionada e que a média amostral
observada foi de 503,24 ml. Esse valor de média amostral é suficientemente maior que 
500 ml para nos fazer rejeitar a hipótese de que a média do processo é 
de 500 ml ao nível de significância de 5%?
"""

amostra = [509, 505, 495, 510, 496, 509, 497, 502, 503, 505, 
           501, 505, 510, 505, 504, 497, 506, 506, 508, 505, 
           497, 504, 500, 498, 506, 496, 508, 497, 503, 501, 
           503, 506, 499, 498, 509, 507, 503, 499, 509, 495, 
           502, 505, 504, 509, 508, 501, 505, 497, 508, 507]

amostra = pd.DataFrame(amostra, columns=['Amostra'])
amostra.head()

media_amostra = amostra.mean()[0]
desvio_padrao_amostra = amostra.std()[0]

media = 500
significancia = 0.05
confianca = 1 - significancia
n = 50

# Passo 1 - formulação das hipóteses H_0 e H_1
# H_0: \mu = 500
# H_1: \mu \neq 500

# Passo 2 - escolha da distribuição amostral adequada
# O tamanho da amostra é maior que 30?
# Resp.: Sim
# O desvio padrão populacional é conhecido?
# Resp.: Não

# Passo 3 - fixação da significância do teste (\alpha)

from scipy.stats import norm
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.weightstats import DescrStatsW

probabilidade = (0.5 + (confianca / 2))
z_alpha_2 = norm.ppf(probabilidade)

# Passo 4 - cálculo da estatística-teste e verificação desse valor com as áreas de aceitação e rejeição do teste
# z = \frac{\bar{x} - \mu_0}{\frac{s}{\sqrt{n}}}

z = (media_amostra - media) / (desvio_padrao_amostra / np.sqrt(n))
print('O valor de z é: ', z)

# Passo 5 - Aceitação ou rejeição da hipótese nula

# Critério do valor crítico
 
# Teste Bicaudal
# Rejeitar H_0 se z \leq -z_{\alpha / 2} ou se z \geq z_{\alpha / 2}

z <= -z_alpha_2

z >= z_alpha_2

# A hipótese nula é rejeitada!

# Critério do $p-valor
 
# Teste Bicaudal
# Rejeitar H_0 se o valor p\leq\alpha

print('Primeira forma de calcular o p-valor: ', 2 * (1 - norm.cdf(z)))
print('Segunda forma de calular o p-valor: ', 2 * (norm.sf(z)))

p_valor <= significancia

# A hipótese nula é rejeitada!

# Outra forma de obter z e o p-valor

valores = ztest(x1 = amostra, value = media)

print('O valor de z é: ', valores[0])
print('O valor de p-valor é: ',valores[1])

test = DescrStatsW(amostra)

z, p_valor = test.ztest_mean(value = media)
print('O valor de z é: ', z[0])
print('O valor de p-valor é: ', p_valor[0])


# Teste Unicaudal

# Os testes unicaudais verificam as variáveis em relação a um piso ou a um teto 
# e avaliam os valores máximos ou mínimos esperados para os parâmetros em estudo 
# e a chance de as estatísticas amostrais serem inferiores ou superiores a dado limite.

# Dados do problema

amostra = [37.27, 36.42, 34.84, 34.60, 37.49, 
           36.53, 35.49, 36.90, 34.52, 37.30, 
           34.99, 36.55, 36.29, 36.06, 37.42, 
           34.47, 36.70, 35.86, 36.80, 36.92, 
           37.04, 36.39, 37.32, 36.64, 35.45]

amostra = pd.DataFrame(amostra, columns=['Amostra'])
amostra.head()

media_amostra = amostra.mean()[0]
desvio_padrao_amostra = amostra.std()[0]

media = 37
significancia = 0.05
confianca = 1 - significancia
n = 25
graus_de_liberdade = n - 1

print('A média da amostra é ', media_amostra.round(2))
print('O desvio padrão da amostra é ', desvio_padrao_amostra.round(2))
print('A significância é ', significancia)
print('A confiança é ', confianca)
print('A quantidade de elementos é ', n)
print('Graus de liberdade ', graus_de_liberdade)


# Passo 1 - formulação das hipóteses H_0 e H_1
# H_0: \mu \leq 37
# H_1: \mu > 37

# Passo 2 - escolha da distribuição amostral adequada
# O tamanho da amostra é maior que 30?
# Resp.: Não

# Podemos afirmar que a população se distribui aproximadamente como uma normal?
# Resp.: Sim
 
# O desvio padrão populacional é conhecido?
# Resp.: Não

# Passo 3 - fixação da significância do teste (\alpha)

from scipy.stats import t as t_student

# Obtendo t_{\alpha}

t_alpha = t_student.ppf(confianca, graus_de_liberdade)

# Passo 4 - cálculo da estatística-teste e verificação desse valor com as áreas de aceitação e rejeição do teste
# t = \frac{\bar{x} - \mu_0}{\frac{s}{\sqrt{n}}}

t = (media_amostra - media) / (desvio_padrao_amostra / np.sqrt(n))

# Critério do valor crítico
 
#  Teste Unicaudal Superior
#  Rejeitar H_0 se t \geq t_{\alpha}

t >= t_alpha


# Critério do valor p
 
# Teste Unicaudal Superior
# Rejeitar H_0 se o valor p\leq\alpha

p_valor = t_student.sf(t, df = 24)
p_valor <= significancia

# Outra forma de obter a resposta

from statsmodels.stats.weightstats import DescrStatsW

test = DescrStatsW(amostra)

t, p_valor, df = test.ttest_mean(value = media, alternative = 'larger')
print('Valor de t', t[0])
print('P-valor', p_valor[0])
print('Média do data frame ', df)

p_valor[0] <= significancia


# Conclusão: Com um nível de confiança de 95% não podemos rejeitar H_0, 
# ou seja, a alegação do fabricante é verdadeira

# Teste Duas Variáveis

# Seleção das amostras

df.head()
df['Species'].unique()

df.shape

# Selecionando as amostras

setosa = df.query('Species == "Iris-setosa"').sample(n = 50, random_state = 10).PetalWidthCm
virginica = df.query('Species == "Iris-virginica"').sample(n = 50, random_state = 10).PetalWidthCm

# Dados do problema

# Conjunto setosa

media_setosa = setosa.mean()
desvio_padrao_setosa = setosa.std()

significancia = 0.01
confianca = 1 - significancia
n_setosa = 50
D_0 = 0

print('Média da amostra: ', media_setosa)
print('Desvio padrão da amostra: ', desvio_padrao_setosa)
print('Significância: ', significancia)
print('Confiança: ', confianca)
print('Quantidade de elementos: ', n_setosa)
print('Graus de liberdade', D_0)

# Conjunto virginica

media_virginica = virginica.mean()
desvio_padrao_virginica = virginica.std()

significancia = 0.01
confianca = 1 - significancia
n_virginica = 50
D_0 = 0

print('Média da amostra: ', media_virginica)
print('Desvio padrão da amostra: ', desvio_padrao_virginica)
print('Significância: ', significancia)
print('Confiança: ', confianca)
print('Quantidade de elementos: ', n_virginica)
print('Graus de liberdade', D_0)


# Passo 1- formulação das hipóteses H_0 e H_1

# \mu_1 \Rightarrow$ Média do tamanho das folhas setosa
# \mu_2 \Rightarrow$ Média tamanho das folhas virginica
# H_0: \mu_1 \leq \mu_2\\
# H_1: \mu_1 > \mu_2
# ou
# H_0: \mu_1 -\mu_2 \leq 0\\
# H_1: \mu_1 -\mu_2 > 0

# Passo 2 - escolha da distribuição amostral adequada

# O tamanho da amostra é maior que 30?
# Resp.: Sim
 
# O desvio padrão populacional é conhecido?
# Resp.: Não

# Passo 3 - fixação da significância do teste (\alpha)

from scipy.stats import norm

probabilidade = confianca
z_alpha = norm.ppf(probabilidade)
z_alpha.round(2)


# Passo 4 - cálculo da estatística-teste e verificação desse valor com as áreas de aceitação e rejeição do teste
# z = \frac{(\bar{x_1} - \bar{x_2})-D_0}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}

numerador = (media_setosa - media_virginica) - D_0
denominador = np.sqrt((desvio_padrao_setosa ** 2 / n_setosa) + (desvio_padrao_virginica ** 2 / n_virginica))
z = numerador / denominador
z

# Passo 5 - Aceitação ou rejeição da hipótese nula

# Critério do valor crítico

# Teste Unicaudal
# Rejeitar H_0 se z \geq z_{\alpha}

z >= z_alpha

# Critério do valor p

# Teste Unicaudal
# Rejeitar H_0 se o valor p\leq\alpha

from statsmodels.stats.weightstats import DescrStatsW, CompareMeans

test_setosa = DescrStatsW(setosa)
test_virginica = DescrStatsW(virginica)
test_A = test_setosa.get_compare(test_virginica)
z, p_valor = test_A.ztest_ind(alternative='larger', value=0)

print('O valor de z é ', z)
print('O p-valor é ', p_valor)

test_B = CompareMeans(test_setosa, test_virginica)
z, p_valor = test_B.ztest_ind(alternative='larger', value=0)

print('O valor de z é ', z)
print('O p-valor é ', p_valor)

p_valor <= significancia

# Aceitamos a hipótese nula.

# Testes não Paramétricos

"""
Teste do Qui-Quadrado (\chi^2)
 
 Também conhecido como teste de adequação ao ajustamento, seu nome se 
 deve ao fato de utilizar uma variável estatística padronizada, representada pela 
 letra grega qui ( \chi) elevada ao quadrado. A tabela com os valores padronizados 
 e como obtê-la podem ser vistos logo abaixo.
 
 O teste do \chi^2 testa a hipótese nula de não haver diferença entre as frequências 
 observadas de um determinado evento e as frequências que são realmente esperadas para este evento.
 
 Os passos de aplicação do teste são bem parecidos aos vistos para os testes paramétricos.
"""

"""
Problema
 Antes de cada partida do campeonato nacional de futebol, as moedas utilizadas pelos árbitros 
 devem ser verificadas para se ter certeza de que não são viciadas, ou seja, que não tendam para 
 determinado resultado. Para isso um teste simples deve ser realizado antes de cada partida. 
 Este teste consiste em lançar a moeda do jogo 50 vezes e contar as frequências de CARAS 
 e COROAS obtidas. A tabela abaixo mostra o resultado obtido no experimento:
 
 ||CARA|COROA|
 |Observado|17|33|
 |Esperado|25|25|
 
 A um nível de significância de 5%, é possível afirmar que a moeda não é honesta, isto é, 
 que a moeda apresenta uma probabilidade maior de cair com a face **CARA** voltada para cima?
"""

#  Dados do problema

F_Observada = [17, 33]
F_Esperada = [25, 25]
significancia = 0.05
confianca = 1 - significancia
k = 2 # Número de eventos possíveis
graus_de_liberdade = k - 1


# Passo 1 - formulação das hipóteses H_0 e H_1
# H_0: F_{CARA} = F_{COROA}
# H_1: F_{CARA} \neq F_{COROA}

# Passo 2 - fixação da significância do teste (\alpha)

from scipy.stats import chi

# chi_{\alpha}^2

chi_2_alpha = chi.ppf(confianca, graus_de_liberdade) ** 2
chi_2_alpha

# Passo 3 - cálculo da estatística-teste e verificação desse valor com as áreas de aceitação e rejeição do teste
# \chi^2 = \sum_{i=1}^{k}{\frac{(F_{i}^{Obs} - F_{i}^{Esp})^2}{F_{i}^{Esp}}}

chi_2 = 0
for i in range(k):
  chi_2 += (F_Observada[i] - F_Esperada[i]) ** 2 /  F_Esperada[i]
 
chi_2

# Passo 4 - Aceitação ou rejeição da hipótese nula
# Critério do valor crítico
# Rejeitar H_0 se \chi_{teste}^2 > \chi_{\alpha}^2

chi_2 > chi_2_alpha

# Critério do valor p
# Rejeitar H_0 se o valor p\leq\alpha

from scipy.stats import chisquare

chi_2, p_valor = chisquare(f_obs=F_Observada, f_exp=F_Esperada)
print('O valor do qui-quadrado é ', chi_2)
print('O valor de p-valor é ', p_valor)

p_valor <= significancia


# Conclusão: Com um nível de confiança de 95% rejeitamos a hipótese nula (H_0) 
# e concluímos que as frequências observadas e esperadas são discrepantes, ou seja, 
# a moeda não é honesta e precisa ser substituída.</font>

"""
Teste Wilcoxon

Comparação de duas populações - amostras dependentes
 
Empregado quando se deseja comparar duas amostras relacionadas, amostras emparelhadas. 
Pode ser aplicado quando se deseja testar a diferença de duas condições, isto é, 
quando um mesmo elemento é submetido a duas medidas.
"""

"""
Problema
 
Um novo tratamento para acabar com o hábito de fumar está sendo empregado em um grupo de 
35 pacientes voluntários. De cada paciente testado foram obtidas as informações de 
quantidades de cigarros consumidos por dia antes e depois do término do tratamento. 
Assumindo um **nível de confiança de 95%** é possível concluir que, depois da aplicação do 
novo tratamento, houve uma mudança no hábito de fumar do grupo de pacientes testado?
"""

# Dados do problema

fumo = {
    'Antes': [39, 25, 24, 50, 13, 52, 21, 29, 10, 22, 50, 15, 36, 39, 52, 48, 24, 15, 40, 41, 17, 12, 21, 49, 14, 55, 46, 22, 28, 23, 37, 17, 31, 49, 49],
    'Depois': [16, 8, 12, 0, 14, 16, 13, 12, 19, 17, 17, 2, 15, 10, 20, 13, 0, 4, 16, 18, 16, 16, 9, 9, 18, 4, 17, 0, 11, 14, 0, 19, 2, 9, 6]
}
significancia = 0.05
confianca = 1 - significancia
n = 35

# Data Frame do problema
fumo = pd.DataFrame(fumo)
fumo.head()

media_antes = fumo.Antes.mean()
media_depois = fumo.Depois.mean()

print('Média de cigarros consumidos antes do tratamento ', media_antes)
print('Média de cigarros consumidos depois do tratamento ', media_depois)


# Passo 1 - formulação das hipóteses H_0 e H_1
# H_0: \mu_{antes} = \mu_{depois}
# H_1: \mu_{antes} > \mu_{depois}

# Passo 2 - escolha da distribuição amostral adequada

# O tamanho da amostra é maior que 20?
# Resp.: Sim

# Passo 3 - fixação da significância do teste ($\alpha$)
# Obtendo z_{\alpha/2}

probabilidade = (0.5 + (confianca / 2))
z_alpha_2 = norm.ppf(probabilidade)
z_alpha_2.round(2) 

# Passo 4 - cálculo da estatística-teste e verificação desse valor com as áreas de aceitação e rejeição do teste
# Z = \frac{T - \mu_T}{\sigma_T}

# Passo 5 - Aceitação ou rejeição da hipótese nula

from scipy.stats import wilcoxon

T, p_valor = wilcoxon(fumo.Antes, fumo.Depois)
print('O valor de T é ', T)
print('O p-valor é ', p_valor)

p_valor <= significancia

# A hipótese nula é rejeitada!

"""
Teste de Mann-Whitney
 
Comparação de duas populações - amostras independentes
 
Mann-Whitney é um teste não paramétrico utilizado para verificar se duas amostras 
independentes foram selecionadas a partir de populações que têm a mesma média. 
Por ser um teste não paramétrico, Mann-Whitney torna-se uma alternativa ao teste 
paramétrico de comparação de médias.
"""

# Selecionando as amostras

setosa = df.query('Species == "Iris-setosa"').sample(n = 50, random_state = 10).PetalWidthCm
virginica = df.query('Species == "Iris-virginica"').sample(n = 50, random_state = 10).PetalWidthCm

# Conjunto setosa

media_setosa = setosa.mean()
desvio_padrao_setosa = setosa.std()

significancia = 0.01
confianca = 1 - significancia
n_setosa = 50
n_1 = len(setosa)

print('Média da amostra: ', media_setosa)
print('Desvio padrão da amostra: ', desvio_padrao_setosa)
print('Significância: ', significancia)
print('Confiança: ', confianca)
print('Quantidade de elementos: ', n_setosa)
print('Graus de liberdade', n_1)

# Conjunto virginica

media_virginica = virginica.mean()
desvio_padrao_virginica = virginica.std()

significancia = 0.01
confianca = 1 - significancia
n_virginica = 50
n_2 = len(virginica)

print('Média da amostra: ', media_virginica)
print('Desvio padrão da amostra: ', desvio_padrao_virginica)
print('Significância: ', significancia)
print('Confiança: ', confianca)
print('Quantidade de elementos: ', n_virginica)
print('Graus de liberdade', n_2)


# Passo 1 - formulação das hipóteses $H_0$ e $H_1$
 
# \mu_1 \Rightarrow Média do tamanho das folhas setosa
# \mu_2 \Rightarrow Média tamanho das folhas virginica

# H_0: \mu_2 = \mu_1\\
# H_1: \mu_2 < \mu_1


# Passo 2 - escolha da distribuição amostral adequada
# Deve-se optar pela distribuição t de Student, já que nada é mencionado sobre a distribuição da 
# população, o desvio padrão populacional é desconhecido e o número de elementos investigados é menor que 30.

# Passo 3 - fixação da significância do teste (\alpha)

from scipy.stats import t as t_student

graus_de_liberdade = n_1 + n_2 - 2
t_alpha = t_student.ppf(significancia, graus_de_liberdade)


# Passo 4 - cálculo da estatística-teste e verificação desse valor com as áreas de aceitação e rejeição do teste

# 1. Definir os n's:
# n_1 = nº de elementos do menor grupo
# n_2 = nº de elementos do maior grupo
# ---
# 2. Obter a soma dos postos
# R_1 = soma dos postos do grupo n_1
# R_2 = soma dos postos do grupo n_2
# ---
# 3. Obter as estatísticas
# u_1 = n_1 \times n_2 + \frac{n_1 \times (n_1 + 1)}{2} - R_1
# u_2 = n_1 \times n_2 + \frac{n_2 \times (n_2 + 1)}{2} - R_2
# ---
# 4. Selecionar o menor U
# u = min(u_1, u_2)
# ---
# 5. Obter a estatística de teste
# Z = \frac{u - \mu{(u)}}{\sigma{(u)}}
# 
# Onde
# 
# \mu{(u)} = \frac{n_1 \times n_2}{2}
# \sigma{(u)} = \sqrt{\frac{n_1 \times n_2 \times (n_1 + n_2 + 1)}{12}}

# Passo 5 - Aceitação ou rejeição da hipótese nula

from scipy.stats import mannwhitneyu

u, p_valor = mannwhitneyu(virginica, setosa, alternative='less')
print('Valor de u ', u)
print('P-valor ', p_valor)

p_valor <= significancia

# A hipótese nula é aceita
