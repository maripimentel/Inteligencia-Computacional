from scipy import signal
import numpy as np
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
import matplotlib.pyplot as plt
import tensorflow as tf

# Cria um dicionario com cada uma das entradas atrasadas
def criaEntradasComAtraso(PATH, vetorAtrasos, entrada, isLbOriginal):
	dicEntradaAtraso = {}

	for numAtrasos in  vetorAtrasos:
		entradaAtraso = np.empty(shape=(numAtrasos+1200,))
		for i in range(1200):
			if (i<numAtrasos):
				for j in range(numAtrasos):
					entradaAtraso[j] = 0

			entradaAtraso[i+numAtrasos] = entrada[i]
		dicEntradaAtraso[numAtrasos] = entradaAtraso

	plotEntradasComAtraso(PATH, dicEntradaAtraso, vetorAtrasos, entrada, isLbOriginal)

	return dicEntradaAtraso

# Gera o grafico contendo as entradas com atraso usadas para teste
def plotEntradasComAtraso(PATH, dicEntradaAtraso, vetorAtrasos, entrada, isLbOriginal):
	plt.figure()

	if isLbOriginal:
		titulo = ' para Largura de Banda [0 0.01]'
	else:
		titulo = ' para Largura de Banda [0 0.05]'

	plt.plot(entrada)
	for numAtrasos in vetorAtrasos:
		plt.plot(dicEntradaAtraso[numAtrasos])
	plt.legend(['Original', 'n = 1', 'n = 5','n = 10'], loc='upper left')
	plt.title('Entrada Original e Entradas Atrasadas'+titulo)
	plt.ylabel('Entrada')
	plt.xlabel('Amostra')
	plt.savefig(PATH+'Entrada/entrada.png')
	plt.grid(True)
	plt.clf()

def adapt(PATH, vetorAprendizagem, vetorAtrasos, saida, saida1, saida2, dicEntradaAtraso):

	print("\n\n")
	print("***********************************")
	print("***************ADAPT***************")
	print("***********************************")

	# Inicializa os dicionario contendo a predicao e o erro para cada taxa de aprendizagem e numero de atrasos
	dicPredicao = {}
	dicErro = {}
	dicAdaline = {}

	for aprendizagem in vetorAprendizagem:
		for numAtrasos in vetorAtrasos:

			# Gera o modelo com um camada Flatten e uma camada Densa Linear
			adaline = Sequential()
			adaline.add(Flatten())
			adaline.add(Dense(1,activation='linear', kernel_initializer='uniform', input_shape=(1,numAtrasos)))
			adaline.compile(loss='mean_squared_error', optimizer='SGD')

			# Inicializa os vetores de predicao, erro e a entrada
			vetorPredicao = []
			vetorErro = []
			entradaAtraso = dicEntradaAtraso[numAtrasos]

			# Erro quadratico inicial é zero
			erroQuadratico = 0.0;

			for i in range(1200):

				# Obtem a entrada da rede
				auxiliar = entradaAtraso[i:numAtrasos+i]

				# Gera a predicao e a armazena no vetor
				entradaAuxiliar = np.reshape(auxiliar,(1,1,numAtrasos))
				predicaoAuxiliar = adaline.predict(entradaAuxiliar)
				predicao = np.reshape(predicaoAuxiliar,(1,))
				vetorPredicao.append(predicao)

				# Calcula o erro e o armazena no vetor
				erro = saida[i] - predicao[0]
				vetorErro.append(erro)

				# Calcula o erro quadratico
				erroQuadratico = erroQuadratico + erro*erro

				# Obtem os pesos da rede
				pesos = adaline.get_weights()

				# Calcula o novo peso conforme a regra delta
				pesosAuxiliar = np.reshape(pesos[0],(numAtrasos,))			
				for j in range(numAtrasos):
					pesosAuxiliar[j] = pesosAuxiliar[j] + aprendizagem*erro*auxiliar[j]
				pesos[0] = np.reshape(pesosAuxiliar,(numAtrasos,1))

				# Configura os novos pesos
				adaline.set_weights(pesos)

			# Imprime o erro quadratico medio para cada valor de taxa de aprendizagem e numero de atrasos
			print("\n\n***********************************")
			print("Aprendizagem: " + str(aprendizagem))
			print("Atraso: " + str(numAtrasos))
			print("Erro Quadratico Medio: " + str(erroQuadratico/1200))
			print("***********************************")

			# Salva os vetores de erro e predicao em dicionarios
			dicPredicao[(aprendizagem,numAtrasos)] = vetorPredicao
			dicErro[(aprendizagem,numAtrasos)] = vetorErro
			dicAdaline[(aprendizagem,numAtrasos)] = adaline

	plotGraficosSaidasErro(PATH+"Adapt/", vetorAprendizagem, vetorAtrasos, dicErro, dicPredicao, saida1, saida2, dicEntradaAtraso, True)

	return (dicAdaline, dicPredicao, dicErro)

def newlind(PATH, vetorAprendizagem, vetorAtrasos, saida, saida1, saida2, dicEntradaAtraso, dicAdaline):

	print("\n\n")
	print("***********************************")
	print("**************NEWLIND**************")
	print("***********************************")

	# Inicializa os dicionario contendo a predicao e o erro para cada taxa de aprendizagem e numero de atrasos
	dicPredicao = {}
	dicErro = {}

	for aprendizagem in vetorAprendizagem:
		for numAtrasos in vetorAtrasos:

			adaline = dicAdaline[(aprendizagem, numAtrasos)]

			# Inicializa os vetores de predicao, erro e a entrada
			vetorPredicao = []
			vetorErro = []
			entradaAtraso = dicEntradaAtraso[numAtrasos]

			# Erro quadratico inicial é zero
			erroQuadratico = 0.0;

			for i in range(1200):

				# Obtem a entrada da rede
				auxiliar = entradaAtraso[i:numAtrasos+i]

				# Gera a predicao e a armazena no vetor
				entradaAuxiliar = np.reshape(auxiliar,(1,1,numAtrasos))
				predicaoAuxiliar = adaline.predict(entradaAuxiliar)
				predicao = np.reshape(predicaoAuxiliar,(1,))
				vetorPredicao.append(predicao)

				# Calcula o erro e o armazena no vetor
				erro = saida[i] - predicao[0]
				vetorErro.append(erro)

				# Calcula o erro quadratico
				erroQuadratico = erroQuadratico + erro*erro

			# Imprime o erro quadratico medio para cada valor de taxa de aprendizagem e numero de atrasos
			print("\n\n***********************************")
			print("Aprendizagem: " + str(aprendizagem))
			print("Atraso: " + str(numAtrasos))
			print("Erro Quadratico Medio: " + str(erroQuadratico/1200))
			print("***********************************")

			# Salva os vetores de erro e predicao em dicionarios
			dicPredicao[(aprendizagem,numAtrasos)] = vetorPredicao
			dicErro[(aprendizagem,numAtrasos)] = vetorErro

	plotGraficosSaidasErro(PATH+"Newlind/", vetorAprendizagem, vetorAtrasos, dicErro, dicPredicao, saida1, saida2, dicEntradaAtraso, False)

	return (dicPredicao, dicErro)


# Gera todos os graficos das saidas e dos erros
def plotGraficosSaidasErro(PATH, vetorAprendizagem, vetorAtrasos, dicErro, dicPredicao, saida1, saida2, dicEntradaAtraso, isAdapt):
	plotGraficosSaidasTotais(PATH, vetorAprendizagem, vetorAtrasos, dicPredicao, saida1, saida2, isAdapt)
	plotGraficosSaidasIndividuais(PATH, vetorAprendizagem, vetorAtrasos, dicPredicao, saida1, saida2, dicEntradaAtraso, isAdapt)
	plotGraficoErros(PATH, vetorAprendizagem, vetorAtrasos, dicErro, isAdapt)

# Gera um grafico com todas as saidas juntas
def plotGraficosSaidasTotais(PATH, vetorAprendizagem, vetorAtrasos, dicPredicao, saida1, saida2, isAdapt):
	fig = plt.figure()

	if(isAdapt):
		fig.suptitle('Saída Esperada e Adapt', fontsize=20)
	else:
		fig.suptitle('Saída Esperada e Newlind', fontsize=20)

	saida = np.concatenate((saida1[1][:], saida2[1][:]))

	for nAprendizagem in range(len(vetorAprendizagem)):
		for nAtrasos in range(len(vetorAtrasos)):
			plt.subplot(331+nAtrasos+len(vetorAtrasos)*nAprendizagem)
			plt.plot(saida, linewidth=1)
			plt.plot(dicPredicao[(vetorAprendizagem[nAprendizagem],vetorAtrasos[nAtrasos])], linewidth=0.8)
			plt.legend(['Original', 'Estimada'], loc='upper left', fontsize=5)
			plt.title('n = '+str(vetorAtrasos[nAtrasos])+ ' lr = ' + str(vetorAprendizagem[nAprendizagem]))
			plt.ylabel('Saída')
			plt.xlabel('Amostra')
			plt.grid(True)

	fig.tight_layout(rect=[0, 0.03, 1, 0.95])	
	plt.savefig(PATH+'predicao.png')
	plt.clf()
	plt.close()

# Gera um grafico para cada saida
def plotGraficosSaidasIndividuais(PATH, vetorAprendizagem, vetorAtrasos, dicPredicao, saida1, saida2, dicEntradaAtraso, isAdapt):
	saida = np.concatenate((saida1[1][:], saida2[1][:]))

	if(isAdapt):
		titulo = 'Saída Esperada e Adapt para '
	else:
		titulo = 'Saída Esperada e Newlind para '

	for nAprendizagem in range(len(vetorAprendizagem)):
		for nAtrasos in range(len(vetorAtrasos)):
			plt.figure()
			plt.title(titulo+'n = '+str(vetorAtrasos[nAtrasos])+ ' lr = ' + str(vetorAprendizagem[nAprendizagem]), fontsize=20)
			plt.plot(saida)
			plt.plot(dicPredicao[(vetorAprendizagem[nAprendizagem],vetorAtrasos[nAtrasos])])
			plt.plot(dicEntradaAtraso[vetorAtrasos[nAtrasos]])
			plt.legend(['Original', 'Estimada', 'Entrada'], loc='upper left')
			plt.ylabel('Valor')
			plt.xlabel('Amostra')
			plt.grid(True)
			plt.savefig(PATH+'predicao_n='+str(vetorAtrasos[nAtrasos])+ '_lr=' + str(vetorAprendizagem[nAprendizagem])+'.png')
			plt.clf()
			plt.close()

# Gera um grafico com todos os erros juntos
def plotGraficoErros(PATH, vetorAprendizagem, vetorAtrasos, dicErro, isAdapt):
	fig = plt.figure()
	
	if(isAdapt):
		fig.suptitle('Erro para a função Adapt', fontsize=20)
	else:
		fig.suptitle('Erro para a função Newlind', fontsize=20)

	for nAprendizagem in range(len(vetorAprendizagem)):
		for nAtrasos in range(len(vetorAtrasos)):
			plt.subplot(331+nAtrasos+len(vetorAtrasos)*nAprendizagem)
			plt.plot(dicErro[(vetorAprendizagem[nAprendizagem],vetorAtrasos[nAtrasos])], linewidth=1)
			plt.title('n = '+str(vetorAtrasos[nAtrasos])+ ' lr = ' + str(vetorAprendizagem[nAprendizagem]))
			plt.ylabel('Erro')
			plt.xlabel('Amostra')
			plt.grid(True)

	fig.tight_layout(rect=[0, 0.03, 1, 0.95])	
	plt.savefig(PATH+'erro.png')
	plt.clf()
	plt.close()


def plotGraficosComparacao(PATH, vetorAprendizagem, vetorAtrasos, dicEntradaAtraso, dicPredicaoAdapt, dicErroAdapt, dicPredicaoNewlind, dicErroNewlind, saida1, saida2):
	PATH = PATH + 'Comparação/';
	plotGraficoErrosComp(PATH, vetorAprendizagem, vetorAtrasos, dicPredicaoAdapt, dicErroAdapt, dicPredicaoNewlind, dicErroNewlind)
	plotGraficoSaidasComp(PATH, vetorAprendizagem, vetorAtrasos, dicEntradaAtraso, dicPredicaoAdapt, dicErroAdapt, dicPredicaoNewlind, dicErroNewlind, saida1, saida2)
	plotGraficosSaidasIndividuaisComp(PATH, vetorAprendizagem, vetorAtrasos, dicEntradaAtraso, dicPredicaoAdapt, dicErroAdapt, dicPredicaoNewlind, dicErroNewlind, saida1, saida2)

# Gera um grafico comparacional dos erros juntos
def plotGraficoErrosComp(PATH, vetorAprendizagem, vetorAtrasos, dicPredicaoAdapt, dicErroAdapt, dicPredicaoNewlind, dicErroNewlind):
	fig = plt.figure()
	fig.suptitle('Erro para as funções Newlind e Adapt', fontsize=20)

	for nAprendizagem in range(len(vetorAprendizagem)):
		for nAtrasos in range(len(vetorAtrasos)):
			plt.subplot(331+nAtrasos+len(vetorAtrasos)*nAprendizagem)
			plt.plot(dicErroAdapt[(vetorAprendizagem[nAprendizagem],vetorAtrasos[nAtrasos])], linewidth=1)
			plt.plot(dicErroNewlind[(vetorAprendizagem[nAprendizagem],vetorAtrasos[nAtrasos])], linewidth=1)
			plt.title('n = '+str(vetorAtrasos[nAtrasos])+ ' lr = ' + str(vetorAprendizagem[nAprendizagem]))
			plt.legend(['Adapt', 'Newlind'], loc='lower left', fontsize=5)
			plt.ylabel('Erro')
			plt.xlabel('Amostra')
			plt.grid(True)

	fig.tight_layout(rect=[0, 0.03, 1, 0.95])	
	plt.savefig(PATH+'erro.png')
	plt.clf()
	plt.close()

# Gera um grafico comparacional das saidas juntos
def plotGraficoSaidasComp(PATH, vetorAprendizagem, vetorAtrasos, dicEntradaAtraso, dicPredicaoAdapt, dicErroAdapt, dicPredicaoNewlind, dicErroNewlind, saida1, saida2):
	fig = plt.figure()
	fig.suptitle('Saída Esperada, Newlind, Adapt e Entrada', fontsize=20)

	saida = np.concatenate((saida1[1][:], saida2[1][:]))

	for nAprendizagem in range(len(vetorAprendizagem)):
		for nAtrasos in range(len(vetorAtrasos)):
			plt.subplot(331+nAtrasos+len(vetorAtrasos)*nAprendizagem)
			plt.plot(saida, linewidth=1)
			plt.plot(dicEntradaAtraso[vetorAtrasos[nAtrasos]], linewidth=1)
			plt.plot(dicPredicaoAdapt[(vetorAprendizagem[nAprendizagem],vetorAtrasos[nAtrasos])], linewidth=0.8)
			plt.plot(dicPredicaoNewlind[(vetorAprendizagem[nAprendizagem],vetorAtrasos[nAtrasos])], linewidth=0.8)
			plt.legend(['Original', 'Entrada','Adapt', 'Newlind'], loc='upper left', fontsize=5)
			plt.title('n = '+str(vetorAtrasos[nAtrasos])+ ' lr = ' + str(vetorAprendizagem[nAprendizagem]))
			plt.ylabel('Saída')
			plt.xlabel('Amostra')
			plt.grid(True)

	fig.tight_layout(rect=[0, 0.03, 1, 0.95])	
	plt.savefig(PATH+'predicao.png')
	plt.clf()
	plt.close()

# Gera um grafico comparacional das saidas individuais
def plotGraficosSaidasIndividuaisComp(PATH, vetorAprendizagem, vetorAtrasos, dicEntradaAtraso, dicPredicaoAdapt, dicErroAdapt, dicPredicaoNewlind, dicErroNewlind, saida1, saida2):
	saida = np.concatenate((saida1[1][:], saida2[1][:]))

	for nAprendizagem in range(len(vetorAprendizagem)):
		for nAtrasos in range(len(vetorAtrasos)):
			plt.figure()
			plt.title('n = '+str(vetorAtrasos[nAtrasos])+ ' lr = ' + str(vetorAprendizagem[nAprendizagem]), fontsize=20)
			plt.plot(saida)
			plt.plot(dicEntradaAtraso[vetorAtrasos[nAtrasos]])
			plt.plot(dicPredicaoAdapt[(vetorAprendizagem[nAprendizagem],vetorAtrasos[nAtrasos])])
			plt.plot(dicPredicaoNewlind[(vetorAprendizagem[nAprendizagem],vetorAtrasos[nAtrasos])])
			plt.legend(['Original', 'Entrada', 'Adapt', 'Newlind'], loc='upper left')
			plt.ylabel('Valor')
			plt.xlabel('Amostra')
			plt.grid(True)
			plt.savefig(PATH+'predicao_n='+str(vetorAtrasos[nAtrasos])+ '_lr=' + str(vetorAprendizagem[nAprendizagem])+'.png')
			plt.clf()
			plt.close()


