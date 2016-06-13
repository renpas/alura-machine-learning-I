from dados import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

x,y = carregar_acessos()
modelo = MultinomialNB()
modelo.fit(x, y)

print(modelo.predict([[1, 0, 1], [0,1,0], [1,0,0],[1,1,0],[1,1,1]]))