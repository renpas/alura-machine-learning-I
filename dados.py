# -*- coding: utf-8 -*-
import csv

def carregar_acessos():
	X = []
	Y = []
	
	arquivo = open('acesso_pagina.csv', 'rb')
	leitor = csv.reader(arquivo)

	#Descarta o cabeçalho
	leitor.next()

	for home,como_funciona,contato,comprou in leitor:
		dado = [int(home), int(como_funciona), int(contato)]
		X.append(dado)
		Y.append(int(comprou))

	return X, Y