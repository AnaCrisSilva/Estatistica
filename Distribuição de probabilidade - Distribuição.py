#!/usr/bin/env python
# coding: utf-8

"""
BIBLIOTECA DE CÓDIGOS
Distribuição de Probabilidade
Ana Cristina Silva - OpusNove

Código em Python 3.6.8
Abril 2021
"""

# Importações

import pandas as pd
import numpy as np

"""
Distribuição Binomial
    Um evento binomial é caracterizado pela possibilidade de ocorrência de apenas duas categorias. 
    Estas categorias somadas representam todo o espaço amostral, sendo também mutuamente excludentes, 
    ou seja, a ocorrência de uma implica na não ocorrência da outra.

    Em análises estatísticas o uso mais comum da distribuição binomial é na solução de problemas 
    que envolvem situações de sucesso e fracasso.

Experimento Binomial
    1. Realização de $n$ ensaios idênticos.
    2. Os ensaios são independentes.
    3. Somente dois resultados são possíveis, exemplo: Verdadeiro ou falso; Cara ou coroa; Sucesso ou fracasso.
    4. A probabilidade de sucesso é representada por p e a de fracasso por 1-p=q. 
       Estas probabilidades não se modificam de ensaio para ensaio.
Média da distribuição binomial
    O valor esperado ou a média da distribuição binomial é igual ao número de experimentos 
    realizados multiplicado pela chance de ocorrência do evento.
Desvio padrão da distribuição binomial
    O desvio padrão é o produto entre o número de experimentos, 
    a probabilidade de sucesso e a probabilidade de fracasso.

-----------------------------------------------------------

Distribuição Poisson

    É empregada para descrever o número de ocorrências em um intervalo de tempo ou espaço específico. 
    Os eventos são caracterizados pela possibilidade de contagem dos sucessos, 
    mas a não possibilidade de contagem dos fracassos. 
    
    Como exemplos de processos onde podemos aplicar a distribuição de Poisson temos a 
    determinação do número de clientes que entram em uma loja em determinada hora, 
    o número de carros que chegam em um drive-thru de uma lanchonete na hora do almoço, 
    a determinação do número de acidentes registrados em um trecho de estrada etc.

Experimento Poisson
    1. A probabilidade de uma ocorrência é a mesma em todo o intervalo observado.
    2. O número de ocorrências em determinado intervalo é independente do número de ocorrências em outros intervalos.
    3. A probabilidade de uma ocorrência é a mesma em intervalos de igual comprimento.

-----------------------------------------------------------

Distribuição Normal

    A distribuição normal é uma das mais utilizadas em estatística. 
    É uma distribuição contínua, onde a distribuição de frequências de uma variável quantitativa 
    apresenta a forma de sino e é simétrica em relação a sua média.

CARACTERÍSTICAS IMPORTANTES:
    1. É simétrica em torno da média;
    2. A área sob a curva corresponde à proporção 1 ou 100%;
    3. As medidas de tendência central (média, mediana e moda) apresentam o mesmo valor;
    4. Os extremos da curva tendem ao infinito em ambas as direções e, teoricamente, jamais tocam o eixo x;
    5. O desvio padrão define o achatamento e largura da distribuição. 
       Curvas mais largas e mais achatadas apresentam valores maiores de desvio padrão;
    6. A distribuição é definida por sua média e desvio padrão; 
    7. A probabilidade sempre será igual à área sob a curva, delimitada pelos limites inferior e superior.


Tabelas padronizadas
        As tabelas padronizadas foram criadas para facilitar a obtenção dos valores das áreas 
        sob a curva normal e eliminar a necessidade de solucionar integrais definidas. 
        Para consultarmos os valores em uma tabela padronizada basta 
        transformarmos nossa variável em uma variável padronizada Z. Esta variável Z representa 
        o afastamento em desvios padrões de um valor da variável original em relação à média.

"""

# combinações de seis números podem ser formadas com os 60 números disponíveis.

from scipy.special import comb

print('Combinações: ', comb(60, 6))

# Probabilidade

print('Probabilidade: ' , '%0.15f' % (1 / comb(60, 6)))

#DISTRIBUIÇÃO BINOMIAL

# Em uma prova que vale 10 pontos e a nota de corte seja 5,
# obtenha a probabilidade deste candidato acertar 5 questões e 
# também a probabilidade deste candidato passar para a próxima etapa do processo seletivo.**

# numero_ensaios 
n = 10
numero_de_alternativas_por_questao = 3

# total_eventos_sucesso 
k = 5

# Probabilidade de sucesso 
p =  1 / numero_de_alternativas_por_questao

# Probabilidade de fracasso
q = 1 - p

from scipy.stats import binom

print('%0.8f' % binom.pmf(k, n, p))

# PROBABILIDADE DO CANDIDATO PASSAR

print('A probabilidade do aluno passar é: ', binom.pmf([5, 6, 7, 8, 9, 10], n, p).sum())
print('A probabilidade do aluno passar é: ', binom.sf(4, n, p))


# DISTRIBUIÇÃO DE POISSON

# Um restaurante recebe em média 20 pedidos por hora. 
#Qual a chance de que, em determinada hora escolhida ao acaso, o restaurante receba 15 pedidos?

# Número médio de pedidos por hora
media = 20

# Número de ocorrências que queremos obter no período ( 𝑘 )
k = 15

from scipy.stats import poisson

print('Probabilidade de receber 15 pedidos: ', '%0.8f' % poisson.pmf(k, media))

# DISTRIBUIÇÃO NORMAL

from scipy.stats import norm


# Em um estudo sobre as alturas dos moradores de uma cidade verificou-se que o 
# conjunto de dados segue uma distribuição aproximadamente normal, com média 1,70 
# e desvio padrão de 0,1. Com estas informações obtenha o seguinte conjunto de probabilidades:

# A.probabilidade de uma pessoa, selecionada ao acaso, ter menos de 1,80 metros.

media = 1.7
desvio_padrao = 0.1
Z = (1.8 - media) / desvio_padrao
norm.cdf(Z)

# B.probabilidade de uma pessoa, selecionada ao acaso, ter entre 1,60 metros e 1,80 metros.    

Z_inferior = (1.6 - media) / desvio_padrao
Z_superior = (1.8 - media) / desvio_padrao
print('A probabilidade é: ',norm.cdf(Z_superior) - (1 - norm.cdf(Z_superior)))
print('A probabilidade é: ', norm.cdf(Z_superior) - norm.cdf(Z_inferior))


# C.probabilidade de uma pessoa, selecionada ao acaso, ter mais de 1,90 metros.

Z = (1.9 - media) / desvio_padrao
print('A probabilidade é: ', 1 - norm.cdf(Z))
print('A probabilidade é: ', norm.cdf(-Z))
