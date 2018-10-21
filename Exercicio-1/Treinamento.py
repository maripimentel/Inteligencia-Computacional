import keras
import tensorflow as tf
import pickle
import cv2
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
import matplotlib.pyplot as plt

tf = [False, True]

for transferenciaDeConhecimento in tf:

	fig = plt.figure()
	modelos = []

	# Percorre todas as redes: RNA0, RNA1, RNA2, RNA3, RNA4, RNA5.
	for rede in range (0,6):

		# Caminho para as pastas de treinamento e de validacao.
		caminhoTreinamento = 'RNA'+str(rede)+'/Treinamento/'
		caminhoValidacao = 'RNA'+str(rede)+'/Validacao/'
		caminhos = {'T':caminhoTreinamento, 'V':caminhoValidacao}

		# Caminho para carregar o modelo e os rotulos.
		caminhoRotulos = 'RNA'+str(rede)+'/rotulos_TL'+str(transferenciaDeConhecimento)+'.dat'
		caminhoModelo = 'RNA'+str(rede)+'/modelo_TL'+str(transferenciaDeConhecimento)+'.hdf5'

		dados = {}

		# Carrega os dados de treinamento e de validacao em um dicionario. 
		for chave in caminhos:

			imagens = []
			rotulos = []

			# Percorre todas as imagens salvas na pasta equivalente a RNA.
			for arquivo in paths.list_images(caminhos[chave]):

				# Le a imagem
				imagem = cv2.imread(arquivo)
				imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

				imagem = np.expand_dims(imagem, axis=2)

				# Pega o rotulo com base no nome da pasta em que a imagem esta salva.
				rotulo = arquivo.split(os.path.sep)[-2]

				# Salva um vetor com as imagens e um com os rotulos.
				imagens.append(imagem)
				rotulos.append(rotulo)

			# Converte os dados para que o Keras possa processa-los.
			imagens = np.array(imagens, dtype="float") / 255.0
			rotulos = np.array(rotulos)

			# Salva os dados em um dicionario.
			# Chaves: XT -> X de Treinamento; YT -> Y de Treinamento;
			# 		  XV -> X de Validacao; YV -> Y de Validacao.
			dados['X'+chave] = imagens
			dados['Y'+chave] = rotulos
			
			if (chave == 'T'):
				binarizador = LabelBinarizer().fit(dados['Y'+chave])

		# Converte os dados gerados pela RNA em rotulos.
		dados['YT'] = binarizador.transform(dados['YT'])
		dados['YV'] = binarizador.transform(dados['YV'])

		with open(caminhoRotulos, "wb") as arquivo:
		    pickle.dump(binarizador, arquivo)

		# Gera os modelos e as camadas.
		modelo = Sequential()

		modelo.add(Flatten())

		# modelo.add(Dense(10000, activation="relu"))

		# modelo.add(Dense(5000, activation="tanh"))

		# modelo.add(Dense(1000, activation="relu"))

		modelo.add(Dense(32, activation="sigmoid"))

		modelo.add(Dense(16, activation="softmax"))

		modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

		if(rede != 0 and transferenciaDeConhecimento):
			modelo.set_weights(pesos)
		

		log = modelo.fit(dados['XT'], dados['YT'], validation_data=(dados['XV'], dados['YV']), batch_size=1, epochs=100, verbose=1)

		modelo.save(caminhoModelo)

		modelos.append(modelo)

		pesos = modelo.get_weights()

		if(transferenciaDeConhecimento):
			fig.suptitle('Acurácia por Epochs com Transferência de Conhecimento', fontsize=16)
		else:
			fig.suptitle('Acurácia por Epochs', fontsize=20)

		plt.subplot(231+rede)
		plt.plot(log.history['acc'])
		plt.plot(log.history['val_acc'])
		plt.title('RNA' + str(rede))
		plt.ylabel('Acurácia')
		plt.xlabel('Epochs')
		plt.grid(True)
		plt.legend(['Treino', 'Teste'], loc='lower right')
	
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])	
	fig.savefig('RNA_Acc_TL'+str(transferenciaDeConhecimento)+'.png')
	plt.clf()
