# -*- coding: utf-8 -*-
#pd = python data_analys
import pandas as pd

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






from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
diferencas = resultado - teste_marcacoes
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_marcacoes)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(taxa_de_acerto)
print(total_de_elementos)