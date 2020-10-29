# -*- coding: utf-8 -*-
"""
"""
# Presenta el programa
import numpy as np
import matplotlib.pyplot as plt


### FUNCIÓN QUE PIDE DATOS AL USUARIO
def PrePar():
    """
    Esta es la función que presenta y pide al usuario
    los parámetros para resolver el programa
    x1 = Inicio del dominio
    x2 = Final del dominio
    N = Número de nodos o número de incógnitas
    L = Longitud todal de la barra de acero (dominio)
    h =
    limx1 = Condición de Dirichlet en x1
    limx2 = Condición de Dirichlet en x2

    Returns
    -------
    Datos.

    """
    print('|---------------------------------------------------|')
    print('| Solución a la ecuación de Transferencia de calor |')
    print('|            Autor: Equpo 1 (azul)                 |' )
    print('| Ingrese los siguientes parámetros de entrada     |')
    print('|--------------------------------------------------|')

    
    a = float(input(' Punto inicial a = '))
    b = float(input(' Punto final b = '))
    N = int(input('Número de nodos N = '))
    L = ( b - a )
    h = L / (N + 1)
    Ta = float(input('Temperatura en a = '))
    Tb = float(input('Temperatura en b = '))
    k = float(input('Valor de la conductividad térmica k =  '))
    Q = []
    tipo = int(input('¿Tu sistema tiene simideros o fuentes? (sí --> 1 ; no --> 2) \n'))
    if tipo == 1:
        for i in range(N):
            Q.append(float(input("Ingresa los valores de Q (presiona enter para poner el siguiente) = ")))
            #Q = float(input('Valor del sumidero o fuente Q = '))
    elif tipo == 2:
        Q = np.zeros(N)
    else:
        print('Número inválido')
    f=int(input('Valor de la diagonal constante f para calibrar = '))
    r = k / (h**2)
    diag = (((h**2)*(f**2)) - 2)
    datos = np.array(11)
    datos = [a,b,N,L,h,Ta,Tb,k,Q,r,diag]
    print('\n Datos = [a,b,N,L,h,Ta,Tb,k,Q,r,diagonal principal]')
    print(datos)
    
    return datos

### FUNCIÓN QUE CONSTRUYE LA MATRIZ A DE TAMAÑO NXN


def MatrizA(N, diag):
    """
    Construye la Matriz A de tamanio NxN

    Parameters
    ----------
    N : Número de nodos o de incógnitas
    diag : el valor de la diagonal principal que depende de la aproximación
    k : valor de la conductividad térmica

    Returns
    -------
    A : TYPE
        DESCRIPTION.

    """    
#    h = N + 1
       
    A = []
    f0 = [diag, 1]
    
    ceros0 = list(map(int, list('0'* (N-2))))
    l0 = f0 + ceros0
    A.append(l0)
    
    cerosm = range(0, N-2)
    f1 = [1, diag, 1]
    
    izquierda = []
    for i in cerosm:
      izquierda.append(list(map(int, list('0'*i))))
    derecha = []
    for i in cerosm[::-1]:
      derecha.append(list(map(int, list('0'*i))))
    
    for i in range(len(izquierda)):
      A.append(izquierda[i] + f1 + derecha[i])
    
    linver = l0[::-1]
    A.append(linver)
    
    #list(map(int,A))
    print('\n La matriz A es:')
    print(np.matrix(A))
    return A 
    

    

### MATRIZ QUE LLENA DE LOS DATOS DE Q

#TENDREMOS QUE FORMAR UNA FUNCION QUE LEA UN ARCHIVO CON LOS VALORES DE Q MÁS ADELANTE


# def MatQ(N,Q):
#     q = [Q]*N
#     print(np.array(q))
#     return np.array(q)

#def MatQ(N,Q):
    # """
    # Parameters
    # ----------
    # N : Tamaño
    # Q : Valor del sumidero o fuente
    # Returns
    # -------
    # Devuelve la matriz q

    # """
 #   q = np.array(N)
    
   # q[0:N]=Q
  #  print('\n La matriz para Q es:')
    
   # print(np.array(q))
   # return q

### MATRIZ DE CONDICIONES DE FRONTERA

def MatDirichlet(N,Ta,Tb):
    """
    
    Parameters
    ----------
    N : Tamaño
    Ta : Condición de frontera de Dirichlet en a
    Tb : Condición de frontera de Dirichlet en b

    Returns
    -------
    Devuelve la matriz f

    """
    f = np.zeros(N)
    f[0] = Ta
    f[N-1] = Tb
    print('\n La matriz para los valores de condiciones de frontera es:')
    print(np.array(f))
    
    return f

### Matriz b

def Matb(Q,f):
    """
    Parameters
    ----------
    q : Matriz Q
    f : Matriz de condiciones de frontera

    Returns
    -------
    b que es una matriz 

    """
    bmat = Q - f
    print('\n  matriz b es:')
    print(np.array(bmat))
    return bmat
    
### Definiendo Matriz solución
def sol(A,b,N, Ta, Tb):
    """
    Parameters
    ----------
    A : Matriz A
    b : Matriz b
    N : Número de nodos
    Ta : Temperatura en la frontera a
    Tb : Temperatura en la frontera b

    Returns
    -------
    u : Matriz solución, sin considerar las temperaturas en las fronteras

    """
    u = np.zeros(N+2)
    u[1:N+1] = np.linalg.solve(A,b)
   # u[0] = Ta
    #u[N+1]=Tb
    print('\n La matriz solución sin condiciones de frontera es:')
    print(np.array(u))
    return u

## Definiendo nueva U para graficar

def u2(u,Ta,Tb, N):
    """
    Parameters
    ----------
    u : Matriz solución sin considerar temperatura en las fronteras
    Ta : Temperatura en la frontera a
    Tb : Temperatura en la frontera b
    N : Número de Nodos

    Returns
    -------
    u2 : Matriz solución con temperatura en las fronteras

    """
      
    u2 = []
    u2 = u
    u2[0] = Ta
    u2[N+1] = Tb
    print('\n La matriz solución con condiciones de frontera es:')

    print(np.array(u2))
    return u2


#def Evalua(x,f):
#    u_orig = ((1-np.cos(f))/(np.sin(f))) * np.sin(f*x) + b * np.cos(f*x)
#    return




# PAra graficar
#def grafica(a,b,N,U):
 #   x = np.linspace(a,b,N+2)
  #  plt.plot(x,U,'-bo')
   # plt.show()



    




