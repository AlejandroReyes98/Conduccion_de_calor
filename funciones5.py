# -*- coding: utf-8 -*-
"""
"""
# Presenta el programa
import numpy as np
import matplotlib.pyplot as plt
#####
"""
V E R S I O N    4 
Modificaciones al 6 de noviembre de 2020
Comentarios
-Se integro una funcion que escoge el problema a resolver
aun no se integra si necesita de diferentes tipos de condiciones 
-Se integro la segunda calibracion con Dirichlet ya 
ingresando los datos de las constantes desde teclado
- Se grafica la solucion analitica y numerica
-La solucion analitica del problema 1 es para el problema normal sin fuentes 
ni sumideros, por eso es la linea recta. No tengo idea de como solucionar los 
otros casos de este ¯\_(ツ)_/¯ 
-Estoy trabajando en la funcion de lectura ggg, principalmente en como hacer 
un formato de archivo general, que seleccione el problema y asi jajaja

"""
#####
### FUNCIÓN QUE PIDE DATOS AL USUARIO
def TipoProblem():
    """
    Funcion que pregunta al usuario que problema se quiere resolver

    Returns
    -------
    Problema que se quiere solucionar.

    """
    print('|--------------------------------------------------|')
    print('|      Solución a la Ecuación de Poisson en 1D     |')
    print('|            Autor: Equipo 1 (azul)                |')
    print('|       ¿Que problema quiere solucionar?           |')
    print('| 1) Transferencia de calor                        |')
    print('| 2) Calibración con condiciones tipo Dirichlet    |')
    print('| 3) Calibración con condicionwa tipo Neumann      |')
    #Falta meter la calibracion 2 gggggg
    print('| Ingrese el número del problema                   |')
    print('|--------------------------------------------------|')
    problema = int(input('Número de problema pro = '))
    return problema

def PrePar(problema):
    """
    Esta es la función que presenta y pide al usuario
    los parámetros para resolver el problema deseado
    x1 = Inicio del dominio
    x2 = Final del dominio
    N = Número de nodos o número de incógnitas
    L = Longitud total de la barra de acero (dominio)
    h = longitud de nodos
    limx1 = Condición de Dirichlet en x1
    limx2 = Condición de Dirichlet en x2

    Returns
    -------
    Datos.

    """
    if problema ==1: 
        print('|--------------------------------------------------|')
        print('| Solución a la ecuación de Transferencia de calor |')
        print('|          en un sistema estacionario.             |')
        print('| Ingrese los siguientes parámetros de entrada     |')
        print('|--------------------------------------------------|')
        a = float(input('| Punto inicial a = '))
        b = float(input('| Punto final b = '))
        N = int(input('| Número de nodos N = '))
        L = ( b - a )
        h = L / (N + 1)
        print("| El tamanio de la malla es      : h = %g " % h)
        Ta = float(input('| Temperatura en a = '))
        Tb = float(input('| Temperatura en b = '))
        k = float(input('| Valor de la conductividad térmica k =  '))
        Q = []
        tipo = int(input('| ¿Tu sistema tiene simideros o fuentes? (sí --> 1 ; no --> 2) \n'))
        if tipo == 1:
            for i in range(N):
                Q.append(float(input("| Ingresa los valores de Q (presiona enter para poner el siguiente) = ")))
    #Q = float(input('Valor del sumidero o fuente Q = '))
        elif tipo == 2:     
            Q = np.zeros(N)
        else:
            print('Número inválido')
        f=int(input('| Valor de la diagonal constante f para calibrar = '))
        r = k / (h**2)
        diag = (((h**2)*(f**2)) - 2)
        datos = np.array(11)
        datos = [a,b,N,L,h,Ta,Tb,k,Q,r,diag]
        print('\n Datos = [a,b,N,L,h,Ta,Tb,k,Q,r,diagonal principal]')
        print(datos)
        return datos
    elif problema==2:
        a = float(input('| Punto inicial       : a = '))
        b = float(input('| Punto final         : b = '))
        # Numero de incognitas y delta x1
        N = int(input('| Numero de nodos     : N = '))
        L=(b-a)
        h = L/(N+1)
        print("| El tamanio de la malla es      : h = %g " % h)
        # Condiciones Dirichlet
        Ta = float(input('| Cond. de front. Dirichlet en a : A = '))
        Tb = float(input('| Cond. de front. Dirichlet en b : B = '))
        #Valores de las constantes 
        cf= float(input('| Valor de constante f : f = '))
        cb= float(input('| Valor de constante b : b = '))
        datos = np.array(9)
        datos=[a,b,N,L,h,Ta,Tb,cf,cb]
        print('\n Datos = [a,b,N,L,h,Ta,Tb,cf,cb]')
        print (datos)
        return datos
    elif problema==3:
        a = float(input('| Punto inicial       : a = '))
        b = float(input('| Punto final         : b = '))
       # Numero de incognitas y delta x1
        N = int(input('| Numero de nodos     : N = '))
        L=(b-a)
        h = L/(N+1)
        print("| El tamanio de la malla es      : h = %g " % h)
        # Condiciones Neumann
        Ta = float(input('| Cond. de front. Neumann en a : A = '))
        Tb = float(input('| Cond. de front. Neumann en b : B = '))
        #Valores de las constantes 
        cf= float(input('| Valor de constante f : f = '))
        cb= float(input('| Valor de constante b : b = '))
        datos = np.array(9)
        datos=[a,b,N,L,h,Ta,Tb,cf,cb]
        print('\n Datos = [a,b,N,L,h,Ta,Tb,cf,cb]')
        print (datos)
        return datos
    else:
        print('Opción no valida')
        exit()        

#TENDREMOS QUE FORMAR UNA FUNCION QUE LEA UN ARCHIVO CON LOS VALORES DE Q MÁS ADELANTE
def ReadF(nombre_de_archivo):
    """
    Lee los parámetros desde un archivo de texto

    Parameters
    ----------
    nombre_de_archivo : Ruta y nombre del archivo que se quiera leer.

    Returns
    -------
    datos: Arreglo con los datos leídos del archivo

    """
    
### FUNCIÓN QUE CONSTRUYE LA MATRIZ A DE TAMAÑO NXN
def MatrizA(N, diag, problema, Ta, Tb):
    """
    Construye la Matriz A de tamanio NxN

    Parameters
    ----------
    N : Número de nodos o de incógnitas
    diag : el valor de la diagonal principal que depende de la aproximación
    k : valor de la conductividad térmica

    Returns
    -------
    A : Matriz A del problema


    """    
#    h = N + 1
       
    A = []
    f0 = [diag, 1]
    
    ceros0 = list(map(int, list('0'* (N-2))))
    l0 = f0 + ceros0
    A.append(l0)
    
    cerosm = range(0, N-2)
    cerosN = [i*0 for i in range(0, N-1)]
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
    
    #Creamos la matriz para el caso de Neumann (AN) cuadno Ta/A = 0
    AN = []
    for i in A[:-1]:
        AN.append(i+[0])  
    AN.append(A[-1]+[1])
    lN = [-1,1]
    AN.append(cerosN + lN)
    
    #CREAMOS LA MATRIZ AN1 PARA EL CASO DE NEUMANN CUANDO Tb/B = 0
    AN1 = []
    linversaN = [1,-1]
    AN1.append(linversaN + cerosN) 
    for i in A[:-1]:
        AN1.append(i+[0])
    #print(AN1)   
    AN1.append(A[-1]+[1])
     
    if problema == 3 and Ta == 0:
        print('\n La matriz A con condiciones de Neumann es: ')
        print(np.matrix(AN))
    elif problema ==3 and Tb == 0:
        print('\n La matriz A con condiciones de Neumann es: ')
        print(np.matrix(AN1))       
    else:
    #list(map(int,A))
        print('\n La matriz A es:')
        print(np.matrix(A))
    return A, AN, AN1


def SolAna(x,prob,d):    
    """
    Funcion que regresa la solución analítica de un problema 
    Parameters
    ----------
    a : integer
        Punto inicial.
    b : integer
        Punto final.
    prob : integer
        Lo que dice que problema se quiere solucionar.

    Returns
    -------
    La solución analítica y el espacio

    """
    #Condicional para selección del problema
    if prob==1:
        return (((d[6]-d[5])/(d[1]-d[0]))*(x-d[1])+d[6])
    elif prob == 2:
        f=d[7]
        b=d[8]
        return (((1-np.cos(f))/np.sin(f))*np.sin(f*x)+b*np.cos(f*x))
    #Esto se descomenta una vez este implementada la opción de calibracion 2
    elif prob == 3:
        return np.exp(x) - x - np.exp(1) + 4
    else:
        print('Opción no valida')
        exit()
### MATRIZ QUE LLENA DE LOS DATOS DE Q


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
    Hace la matriz para condiciones de Dirichlet
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
    Hace la matriz B
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
def sol(A,b,N):
    """
    Soluciona el sistema de ecuaciones
    Parameters
    ----------
    A : Matriz A
    b : Matriz b
    N : Número de nodos

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

#MATRIZN CON PROBLEMAS DE FRONTERA TIPO NEUMANN I
def MatNeumman(N,Ta,Tb,h):
    """
    Hace la matriz para condiciones de Neumann
    Parameters
    ----------
    N : Tamaño
    Ta : Condición de frontera de Neumann en a
    Tb : Condición de frontera de Neumann en b

    Returns
    -------
    Devuelve la matriz f

    """
    z = np.zeros(N+2)
    z[0] = -Ta
    #N-2 si no funciona N
    z[-1] =- h * Tb
    print('\n La matriz para los valores de condiciones de frontera es:')
    print(np.array(z))
    
    return z

def MatbN(Q1,z,N):
    """
    Hace la matriz B
    Parameters
    ----------
    q : Matriz Q
    z : Matriz de condiciones de frontera

    Returns
    -------
    b que es una matriz 

    """
    Q1 = np.zeros[N+2]
    print(Q1)
    print(z)
    bmatN = Q1 + z
    print('\n  matriz b es:')
    print(np.array(bmatN))
    return bmatN

### Definiendo Matriz solución para Neumann
def solN(AN,AN1,bmatN,N,Ta,Tb):
    """
    Soluciona el sistema de ecuaciones
    Parameters
    ----------
    AN, AN1 : Matriz A para el caso de Neumann
    b : Matriz b
    N : Número de nodos

    Returns
    -------
    u : Matriz solución, sin considerar las temperaturas en las fronteras

    """
    u = np.zeros(N+2)
    u[1:N+1] = np.linalg.solve(AN,bmatN)
    u[0] = Ta
    u[-1]= Tb
    print('\n La matriz solución sin condiciones de frontera es:')
    print(np.array(u))
    return u
    

#def Evalua(x,f):
#    u_orig = ((1-np.cos(f))/(np.sin(f))) * np.sin(f*x) + b * np.cos(f*x)
#    return




# PAra graficar
#def grafica(a,b,N,U):
 #   x = np.linspace(a,b,N+2)
  #  plt.plot(x,U,'-bo')
   # plt.show()
