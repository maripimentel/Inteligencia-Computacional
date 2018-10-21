import pickle
import numpy as np
from imutils import paths
from keras.models import load_model
import cv2
import os.path
import matplotlib.pyplot as plt

tf = [False, True]

for transferenciaDeConhecimento in tf:

	# Percorre todas as redes: RNA0, RNA1, RNA2, RNA3, RNA4, RNA5
	for rede in range (0,6):

		# Caminho para carregar o modelo e os rotulos.
		caminhoRotulos = 'RNA'+str(rede)+'/rotulos_TL'+str(transferenciaDeConhecimento)+'.dat'
		caminhoModelo = 'RNA'+str(rede)+'/modelo_TL'+str(transferenciaDeConhecimento)+'.hdf5'

		# Caminho para as pastas de validacao.
		caminhoValidacao = 'RNA'+str(rede)+'/Validacao/'

		# Abre o arquivo de rotulos para escrita.
		with open(caminhoRotulos, "rb") as arquivo:
		    arquivoRotulos = pickle.load(arquivo)

		modelo = load_model(caminhoModelo)

		verdadeiro = [0,0,0,0,0,0]
		falso = [0,0,0,0,0,0]
		porcentagem = [0.0,0.0,0.0,0.0,0.0,0.0]

		for arquivo in paths.list_images(caminhoValidacao):

			imagemOriginal = cv2.imread(arquivo)
			imagem = cv2.cvtColor(imagemOriginal, cv2.COLOR_BGR2GRAY)

			imagem = np.expand_dims(imagem, axis=2)
			imagem = np.expand_dims(imagem, axis=0)

			rotulo = arquivo.split(os.path.sep)[-2]

			predicao = modelo.predict(imagem)
			resultado = arquivoRotulos.inverse_transform(predicao)[0]

			ruido = int(arquivo[22])

			if (resultado == rotulo):
				verdadeiro[ruido] = verdadeiro[ruido] + 1
			else:
				falso[ruido] = falso[ruido] + 1

		for indice in range(0, 6):
			porcentagem[indice] = float(100*float(verdadeiro[indice])/(float(verdadeiro[indice])+float(falso[indice])))

		ruidoGerado = [0,10,20,30,40,50]

		print("Rede: "+str(rede))
		print("Verdadeiros: "+str(verdadeiro))
		print("Falso: "+str(falso))
		print("Final: "+str(porcentagem))

		plt.plot(ruidoGerado, porcentagem)
		if(transferenciaDeConhecimento):
			plt.title('Acurácia por Ruído com Transferência de Conhecimento', fontsize=16)
		else:
			plt.title('Acurácia por Ruído', fontsize=20)
		
		plt.ylabel('Acurácia')
		plt.xlabel('Ruído')
		plt.grid(True)
		
	plt.legend(['RNA0', 'RNA1', 'RNA2', 'RNA3', 'RNA4', 'RNA5'], loc='upper right')
	plt.savefig('RNA_AccRuido_TL'+str(transferenciaDeConhecimento)+'.png')
	plt.clf()