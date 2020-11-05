# -*- coding: utf-8 -*-
"""
Programa
"""
# Programa principal
# Se llaman las funciones necesarias
import numpy as np
import funcionesv3 as f
import matplotlib.pyplot as plt

#Leemos los parametros de entrada:
datos = f.PrePar()

# Se define nuestro sistema Lineal: Matriz A
N = datos[2]
a = datos[0]
b = datos[1]
k = datos[7]
Q = datos[8]
r = datos[9]
Ta = datos[5]
Tb = datos[6]
diag = datos[10]
#MatrizA(N,diag,k)
A = f.MatrizA(N, (diag))

#Se define la matríz Q (Este va a depender, en este caso será cero, pero se definirá)
#MatrizQ = f.MatQ(N, Q)


print('Matriz Q')
print(Q)
#
MatrizQ = np.asarray(Q) / r 

MatCondiciones = f.MatDirichlet(N,Ta,Tb)

Matb = f.Matb(MatrizQ,MatCondiciones)
#print(np.array(b))
solucion = f.sol(A,Matb,N)
#Solucion analítica:
sa=f.SolAna(a,b)
#print(solucion)
U = f.u2(solucion,Ta,Tb, N)
x = np.linspace(a,b,N+2)

#print(x)
#print(len(U))



#GRAFICA
plt.plot(x,U,'or' )
plt.rcParams["figure.figsize"] = (5, 1)
plt.xlabel('Dominio')
plt.ylabel('Temperatura')
plt.title('Conducción de calor')
#plt.set_cmap('hot')
plt.show()
#graf = grafica(int(datos[0]), int(datos[1]), N, U)
#print(np.matrix(A))

