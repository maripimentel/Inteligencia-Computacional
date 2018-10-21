from GeraConjuntoDeDados import *

# Gera o alfabeto original e salva as imagens
alfabeto = geraAlfabeto()
alfabetos = {0:alfabeto}
salvaImagens(alfabeto, 0, 0, True)

# Gera o alfabeto com 10, 20, 30, 40 e 50 porcento de ruido e salva as imagens
for i in range(10, 50 + 1, 10):

	# 10 amostras de cada alfabeto com ruido para treinamento
	for j in range (0,10):
		alfabeto = geraAlfabeto()
		alfabetos["Ruido:"+str(i)+" Amostra:"+str(j)] = acrescentaRuido(alfabeto, i)
		salvaImagens(alfabetos["Ruido:"+str(i)+" Amostra:"+str(j)], i, j, True)


for i in range(0, 50 + 1, 10):

	if(i == 0):
		numeroAmostras = 1
	else:
		numeroAmostras = 10

	# 10 amostras de cada alfabeto com ruido para teste (distintas das amostras para treino)
	for j in range (0,numeroAmostras):
		alfabeto = geraAlfabeto()
		alfabetos["Ruido:"+str(i)+" Amostra:"+str(j)] = acrescentaRuido(alfabeto, i)
		salvaImagens(alfabetos["Ruido:"+str(i)+" Amostra:"+str(j)], i, j, False)


