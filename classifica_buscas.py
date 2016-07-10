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


porcentagem_de_treino = 0.9
tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_teste  = len(Y) - tamanho_de_treino

treino_dados     = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados      = X[-tamanho_de_teste:]
teste_marcacoes  = Y[-tamanho_de_teste:]






#from sklearn.naive_bayes import MultinomialNB
#modelo = MultinomialNB()

from sklearn.ensemble import AdaBoostClassifier
modelo = AdaBoostClassifier()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
#acertos = sum(resultado == teste_marcacoes)
#acertos = [d for d in diferencas if d == 0]
total_de_acertos = sum(resultado == teste_marcacoes)
total_de_elementos = len(teste_marcacoes)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print("Taxa de acerto do algoritmo: %.2f" % taxa_de_acerto)
print("Total de elementos %d " % total_de_elementos)

# a eficacia do algoritmo que chuta 
# tudo um unico valor
acerto_base = max(Counter(teste_marcacoes).itervalues())
taxa_de_acerto_base = 100.0 * acerto_base / len(teste_marcacoes)
print("Taxa de acerto base: %.2f" % taxa_de_acerto_base)