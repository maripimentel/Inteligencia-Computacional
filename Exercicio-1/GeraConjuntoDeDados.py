import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import randint
import os, sys
from pathlib import Path

def geraAlfabeto():

	# Matrizes correspondente a cada letra do M ao B.
	M =[[1,0,0,0,0,0,1],
		[1,1,0,0,0,1,1],
		[1,0,1,0,1,0,1],
		[1,0,0,1,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1]]

	N =[[1,0,0,0,0,0,1],
		[1,1,0,0,0,0,1],
		[1,1,1,0,0,0,1],
		[1,0,1,1,0,0,1],
		[1,0,0,1,1,0,1],
		[1,0,0,0,1,1,1],
		[1,0,0,0,0,1,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1]]

	O =[[1,1,1,1,1,1,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,1,1,1,1,1,1]]

	P =[[1,1,1,1,1,1,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,1,1,1,1,1,1],
		[1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0]]

	Q =[[1,1,1,1,1,1,0],
		[1,0,0,0,0,1,0],
		[1,0,0,0,0,1,0],
		[1,0,0,0,0,1,0],
		[1,0,0,0,0,1,0],
		[1,0,0,1,0,1,0],
		[1,0,0,0,1,1,0],
		[1,1,1,1,1,1,0],
		[0,0,0,0,0,0,1]]

	R =[[1,1,1,1,1,1,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,1,1,1,1,1,1],
		[1,1,0,0,0,0,0],
		[1,0,1,0,0,0,0],
		[1,0,0,1,0,0,0],
		[1,0,0,0,1,0,0]]

	S =[[1,1,1,1,1,1,1],
		[1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0],
		[1,1,1,1,1,1,1],
		[0,0,0,0,0,0,1],
		[0,0,0,0,0,0,1],
		[0,0,0,0,0,0,1],
		[1,1,1,1,1,1,1]]

	T =[[1,1,1,1,1,1,1],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0]]

	U =[[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,1,1,1,1,1,1]]


	V =[[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[0,1,0,0,0,1,0],
		[0,1,0,0,0,1,0],
		[0,0,1,0,1,0,0],
		[0,0,1,0,1,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0]]

	W =[[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,1,0,0,1],
		[1,0,1,0,1,0,1],
		[1,1,0,0,0,1,1],
		[1,0,0,0,0,0,1]]

	X =[[1,0,0,0,0,0,1],
		[1,1,0,0,0,1,1],
		[0,1,1,0,1,1,0],
		[0,0,1,1,1,0,0],
		[0,0,0,1,0,0,0],
		[0,0,1,1,1,0,0],
		[0,1,1,0,1,1,0],
		[1,1,0,0,0,1,1],
		[1,0,0,0,0,0,1]]

	Y =[[1,0,0,0,0,0,1],
		[1,1,0,0,0,1,1],
		[0,1,1,0,1,1,0],
		[0,0,1,1,1,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,1,0,0,0]]

	Z =[[1,1,1,1,1,1,1],
		[0,0,0,0,0,0,1],
		[0,0,0,0,0,1,0],
		[0,0,0,0,1,0,0],
		[0,0,0,1,0,0,0],
		[0,0,1,0,0,0,0],
		[0,1,0,0,0,0,0],
		[1,0,0,0,0,0,0],
		[1,1,1,1,1,1,1]]

	A =[[1,1,1,1,1,1,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,1,1,1,1,1,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,0,1]]

	B =[[1,1,1,1,1,0,0],
		[1,0,0,0,0,1,0],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,1,0],
		[1,1,1,1,1,0,0],
		[1,0,0,0,0,1,0],
		[1,0,0,0,0,0,1],
		[1,0,0,0,0,1,0],
		[1,1,1,1,1,0,0]]

	# Criacao de um dicionario para armazenar cada uma das matrizes criadas e a letra correspondente.
	alfabeto = {"M":M, "N":N, "O":O, "P":P, "Q":Q, "R":R, "S":S, "T":T, "U":U, "V":V, "W":W, "X":X, "Y":Y, "Z":Z, "A":A, "B":B}

	return alfabeto

def salvaImagens(alfabeto, ruido, amostra, treinamento):

	if (treinamento):
		redes = range((int)(ruido/10), 6)
	else:
		redes = range(0,6)

	for rede in redes:

		caminho = 'RNA'+str(rede)

		if Path(caminho).is_dir()==False:
			os.mkdir(caminho);

		if (treinamento):
			caminho = caminho+'/Treinamento'
		else:
			caminho = caminho+'/Validacao'

		if Path(caminho).is_dir()==False:
			os.mkdir(caminho);

		# Salva uma imagem para cada letra.
		for key in alfabeto:
			if Path(caminho+'/'+key).is_dir()==False:
				os.mkdir(caminho+'/'+key);
			plt.imsave(caminho+'/'+key+'/ruido'+str(ruido)+'amostra'+str(amostra)+'.png', np.array(alfabeto[key]), cmap=cm.gray)

def geraRuido(letra, ruido):

	# Calcula o numero de pixels equivalente a porcentagem de ruido.
	numeroPixelsRuido = (int) (9*7*ruido/100)

	# historico = []

	# Percorre os pixels mudando o valor de posicoes aleatorias.
	for i in range (0,numeroPixelsRuido):
		
		linha = randint(0, 8)
		coluna = randint(0, 6)

		# # Verifica se a posicao nao foi alterada anteriormente
		# while ([linha, coluna] in historico):
		# 	linha = randint(0, 8)
		# 	coluna = randint(0, 6)
			

		# # Mantem o historico dos pixels alterados.
		# historico.append([linha,coluna])

		# Troca o valor do pixels
		if (letra[linha][coluna] == 1):
			letra[linha][coluna] = 0
		else:
			letra[linha][coluna] = 1

	return letra

def acrescentaRuido(alfabeto, ruido):

	alfabetoRuido = {}

	# Percorre cada letra do alfabeto, calculando a letra com ruido
	for key in alfabeto:
		letra = geraRuido(alfabeto[key], ruido)
		alfabetoRuido[key] = letra

	return alfabetoRuido
