from scipy import signal
import numpy as np
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
import matplotlib.pyplot as plt
import tensorflow as tf
from bib import *

FREQ = 'freq=001'
PATH = 'Gráficos/'+FREQ+'/'

if FREQ == 'freq=001':
	isLbOriginal = True

# Discretizacao das funcoes de tranferencias
numerador = [1]
denominador = [1,0.2,1]

gz1 = signal.cont2discrete((numerador, denominador), 0.1, 'zoh')

numerador = [3]
denominador = [1,2,1]

gz2 = signal.cont2discrete((numerador, denominador), 0.1, 'zoh')

# Sinal de entrada lido de um arquivo csv gerado por um codigo Matlab
sinalEntrada = np.genfromtxt('dados001.csv',float)

# Definicao da entrada e do tempo dividida para cada funcao de transferencia
entrada1 = sinalEntrada[0:800,]
entrada2 = sinalEntrada[800:1201,]
entrada = sinalEntrada

tempo1 = np.linspace(0.0,80.0,num=800)
tempo2 = np.linspace(0.0,40.0,num=400)
tempo = np.linspace(0.0,120.0,num=1200)

# Calculo das saidas
saida1 = signal.dlsim(gz1, entrada1, tempo1)
saida2 = signal.dlsim(gz2, entrada2, tempo2)
saidaPlt = np.concatenate((saida1[1][:], saida2[1][:]))
saida = np.concatenate((saida1[1][:], saida2[1][:]))

# Taxa de aprendizagem
vetorAprendizagem = [0.05, 0.1, 0.2]

# Numero de Atrasos
vetorAtrasos = [1, 5, 10]

# Cria a entrada para cada valor de atraso
dicEntradaAtraso = criaEntradasComAtraso(PATH, vetorAtrasos, entrada, isLbOriginal)

# Funcao ADAPT
(dicAdaline, dicPredicaoAdapt, dicErroAdapt) = adapt(PATH, vetorAprendizagem, vetorAtrasos, saida, saida1, saida2, dicEntradaAtraso)

# Funcao NEWLIND
(dicPredicaoNewlind, dicErroNewlind) = newlind(PATH, vetorAprendizagem, vetorAtrasos, saida, saida1, saida2, dicEntradaAtraso, dicAdaline)

# Graficos para comparacao dos resultados
plotGraficosComparacao(PATH, vetorAprendizagem, vetorAtrasos, dicEntradaAtraso, dicPredicaoAdapt, dicErroAdapt, dicPredicaoNewlind, dicErroNewlind, saida1, saida2)
