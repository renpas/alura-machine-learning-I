# -*- coding: utf-8 -*-
#pd = python data_analys
import pandas as pd
from collections import Counter

#df = data_frame
df = pd.read_csv('busca.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

#as type garante que o valor vai ser do tipo inteiro
Xdummies_df = pd.get_dummies(X_df).astype(int)
#Y não precisa ser transformado
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values


porcentagem_de_treino = 0.8
porcentagem_de_teste  = 0.1


tamanho_de_treino    = int(porcentagem_de_treino * len(Y))
tamanho_de_teste     = int(porcentagem_de_teste  * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

treino_dados     = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

fim_de_treino    = tamanho_de_treino + tamanho_de_teste
teste_dados      = X[tamanho_de_treino: fim_de_treino]
teste_marcacoes  = Y[tamanho_de_treino: fim_de_treino]

validacao_dados     = X[fim_de_treino:]
validacao_marcacoes = Y[fim_de_treino:]

def fit_and_predict(modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
	modelo.fit(treino_dados, treino_marcacoes)
	resultado = modelo.predict(teste_dados)
	total_de_acertos = sum(resultado == teste_marcacoes)
	total_de_elementos = len(teste_marcacoes)

	taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

	nome = type(modelo).__name__
	msg  = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
	print(msg)
	return taxa_de_acerto


from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict(modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)


from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict(modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if resultadoMultinomial > resultadoAdaBoost:
	vencendor = modeloMultinomial
else:
	vencendor = modeloAdaBoost



resultado = vencendor.predict(validacao_dados)
total_de_acertos = sum(resultado == validacao_marcacoes)
total_de_elementos = len(validacao_marcacoes)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

nome = type(vencendor).__name__
msg  = "Taxa de acerto do {0} no mundo real: {1}".format(nome, taxa_de_acerto)
print(msg)








# a eficacia do algoritmo que chuta 
# tudo um unico valor
acerto_base = max(Counter(validacao_marcacoes).itervalues())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %.2f" % taxa_de_acerto_base)

print("Total de teste %d " % len(validacao_marcacoes))