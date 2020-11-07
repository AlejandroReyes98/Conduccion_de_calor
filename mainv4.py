# -*- coding: utf-8 -*-
"""
Programa
V E R S I O N   4 
Comentarios:
-Comentarios mas detallados estan en funciones
-Implemente la primer calibracion (づ￣ ³￣)づ
-Peeeero, es necesario mover los ifs de aqui a funciones 
-Quiza tengamos una funcion de mas, creo que podemos juntar la funcion que llena
la matriz Q con la matriz b.

"""
# Programa principal
# Se llaman las funciones necesarias
import numpy as np
import funcionesv4 as f
import matplotlib.pyplot as plt

#Escogemos el tipo de problema
pro=f.TipoProblem()
#Leemos los parametros de entrada:
datos = f.PrePar(pro)

# Se define nuestro sistema Lineal: Matriz A
a = datos[0]
b = datos[1]
N = datos[2]
h=datos[4]
Ta = datos[5]
Tb = datos[6]
if pro==1:    
    diag = datos[10]
    r = datos[9]
    Q = datos[8]
    k = datos[7]
    cf=0
    cb=0
elif pro==2:
    cf=datos[7]
    cb=datos[8]

#MatrizA(N,diag,k)
#A = f.MatrizA(N, (diag))#Esto lo pueden descomentar, creo que para Condiciones Neumann es lo que se hara
A = f.MatrizA(N, -2)

#Se define la matríz Q (Este va a depender, en este caso será cero, pero se definirá)
#MatrizQ = f.MatQ(N, Q)
x= np.linspace(a,b,N+2)
#Problema 1
if pro==1:
    print('Matriz Q')
    print(Q)
    MatrizQ = np.asarray(Q) / r 
    MatCondiciones = f.MatDirichlet(N,Ta,Tb)
    Matb = f.Matb(MatrizQ,MatCondiciones)
#print(np.array(b))
#Problema 2
elif pro==2:
    #Esto se podria implementar en una funcion, esta aqui no mas de prueba
    Matb = np.zeros(N)         # Lado derecho             
    Matb= -(cf**2)*h*h*((((1-np.cos(cf))/np.sin(cf))*np.sin(cf*x[1:N+1])+cb*np.cos(cf*x[1:N+1]))) 
    Matb[0] -= Ta
    Matb[N-1] -= Tb
#Parte que soluciona la matriz   
solucion = f.sol(A,Matb,N)
U = f.u2(solucion,Ta,Tb, N)    

#Solucion analítica:
xs=np.linspace(a,b,100)
sa=f.SolAna(xs,pro,datos)
#print(x)
#print(len(U))



#GRAFICA
plt.plot(x,U,'-or', label='Solucion numérica')
plt.plot(xs,sa,'-b',label='Solucion analítica')
#plt.rcParams["figure.figsize"] = (5, 1)
plt.xlabel('Dominio')
plt.ylabel('Temperatura')
plt.title('Solucion')
plt.legend(loc='upper right')
#plt.set_cmap('hot')
plt.show()
#graf = grafica(int(datos[0]), int(datos[1]), N, U)
#print(np.matrix(A))

