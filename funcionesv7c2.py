# -*- coding: utf-8 -*-
"""
"""
# Presenta el programa
import numpy as np
import matplotlib.pyplot as plt
import os 
#####
"""
V E R S I O N   6        12/11/2020
-Se separan las funciones por cada calibración, problema, etc...
-Se implementaron las funciones para leer archivos en este caso solo he integrado
para el primer problema
    -IMPORTANTE: El archivo con el que probe es PARAMETROS
    Tiene este formato:
PuntoInicial PuntoFinal Nodos CondicionDirchA CondicionDirchB K Q
(Son ESPACIOS, esto es importante)    
-Como van a ser programas principales diferentes para cada problema
-Inserte una funcion de Titulo gg
-Se agrego una funcion que hace lo de np.linspace jajaja pero como intento solo usar
los valores del vector de datos entonces agarra los valores de a,b y N del vector
para hacer el dominio numerico, ya que su posicion en todos los problemas será igual
-Lo mismo se hace con el dominio analitico
-Se integro una funcion que grafica la solución Numérica vs Analítica
-Se integro la funcion que calcula el error
I    M    P   O   R   T   A   N   T   E:
-Este es el vector d o datos:
        d: Vector de datos con las siguientes posiciones.
        En todos los problemas y calibraciones:
        d[0]: Punto inicial a.
        d[1]: Punto final b.
        d[2]: Numero de nodos. 
        d[3]: Longitud del segmento L.
        d[4]: Tamanio de la malla h.
        
         En el problema 1:
        d[5]: Condición Dirichlet en A.
        d[6]: Condición Dirichlet en B.
        d[7]: Valor de k.
        d[8]: Valor de q(Escalar). 
        d[9]: Valor de -k / (h**2).
        
       En la calibración 1:
        d[5]: Condición Dirichlet en A.
        d[6]: Condición Dirichlet en B.
        d[7]: El valor de f (influye en la diagonal principal de la matriz A).
        d[8]: El valor b de la solución analítica. Como lo desconocemos b=1
        
        En la calibración 2:
        d[5]: Condición Neumann en A.
        d[6]: Condición Neumann en B.
        d[7]: El valor de f (influye en la diagonal principal de la matriz A).
        d[8]: El valor b de la solución analítica. Como lo desconocemos b=1
        
"""
#####
### FUNCIÓN QUE PIDE DATOS AL USUARIO
def Titulo():
    print('|--------------------------------------------------|')
    print('|      Solución a la Ecuación de Poisson en 1D     |')
    print('|            Autor: Equipo 1 (azul)                |')
    print('|--------------------------------------------------|')
    
def PrePar1():
    """
    Esta es la función que presenta y pide al usuario
    los parámetros para resolver el problema deseado
    x1 = Inicio del dominio
    x2 = Final del dominio
    N = Número de nodos o número de incógnitas
    L = Longitud total de la barra de acero (dominio)
    h = longitud de nodos
    limx1 = Condición de Neumann en x1
    limx2 = Condición de Neumann en x2

    Returns
    -------
    Vector Datos con los datos en las siguientes posiciones:
        

    """
    
    print('|--------------------------------------------------|')
    print('|      Solución a la Ecuación de Poisson en 1D     |')
    print('|            Autor: Equipo 1 (azul)                |')
    print('|--------------------------------------------------|')
    print('| Solución a la ecuación de Transferencia de calor |')
    print('|          con condiciones tipo Neumann            |')
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
    #Q = []
    r = -k / (h**2)
    tipo = int(input('| ¿Tu sistema tiene simideros o fuentes? (sí --> 1 ; no --> 2) \n'))
    if tipo == 1:
            q=(float(input("| Ingresa el valor de Q (presiona enter para poner el siguiente) = ")))
            #Q=(np.ones(N)*q)/r
    #Q = float(input('Valor del sumidero o fuente Q = '))
    elif tipo == 2:     
        #Q = np.zeros(N)
        q=0
    else:
        print('Número inválido')
    #f=int(input('| Valor de la diagonal constante f para calibrar = '))
    
    #diag = (((h**2)*(f**2)) - 2)
    datos = np.array(10)
    datos = [a,b,N,L,h,Ta,Tb,k,q,r]
    print('\n Datos = [a,b,N,L,h,Ta,Tb,k,q,r]')
    print(datos)
    return datos
 #################################################   

def PrePar2():
    """
    Esta es la función que presenta y pide al usuario
    los parámetros para resolver la calibración 1
    
    a = Inicio del dominio
    b = Final del dominio
    N = Número de nodos o número de incógnitas
    L = Longitud total de la barra de acero (dominio)
    h = longitud de nodos
    Ta = Condición de Dirichlet en a
    Tb = Condición de Dirichlet en b
    cf =  valor de f de la calibración
    cb = valor b de la solución analítica, al desconocerla será 1

    Returns
    -------
    datos : Vector con todos los datos introducidos por el usuario

    """
    
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
    #cb = float(1)
    datos = np.array(9)
    datos=[a,b,N,L,h,Ta,Tb,cf,cb]
    print('\n Datos = [a,b,N,L,h,Ta,Tb, f,b calibración]')
    print (datos)
    return datos


def PrePar3():
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


      
#TENDREMOS QUE FORMAR UNA FUNCION QUE LEA UN ARCHIVO CON LOS VALORES DE Q MÁS ADELANTE
def Mostrar():
    """
    Muestra lo que hay en el directorio actual

    """
    b=os.listdir()
    print(b)
    print('|--------------------------------------------------|')

def ReadF1(archivo):
    """
    Lee los parámetros desde un archivo de texto

    Parameters
    ----------
    archivo : Ruta y nombre del archivo que se quiera leer.

    Returns
    -------
    datos: Arreglo con los datos leídos del archivo

    """
    if os.path.exists(archivo):#Verifica si el archivo existe
        ifile = open(archivo, 'r')     # abre el archivo de entrada
        file_lines = ifile.readlines()  # lee las lineas del archivo
        ifile.close();                  # cierra el archivo de entrada
        a, b, N, A, B,k,q = file_lines[0].split() # separa las columnas de la primera linea
        a = float(a); b = float(b); N = int(N); A = float(A); B = float(B); k=float(k); q=float(q) # convierte los datos
        L = ( b - a )
        h = L / (N + 1)
        r = -k / (h**2)
        datos = np.array(10)
        datos = [a,b,N,L,h,A,B,k,q,r]
        print('\n Datos = [a,b,N,L,h,Ta,Tb,k,q,r]')
        print(datos)
    else:
        print('No existe el archivo')
        datos=[]
    return datos

 #################################################   

#Función que lee datos para la CALIBRACION 1

def ReadF2(archivo):
    """
    Lee los parámetros desde un archivo de texto para la calibración 1

    Parameters
    ----------
    archivo : Ruta y nombre del archivo que se quiera leer.

    Returns
    -------
    datos: Arreglo con los datos leídos del archivo

    """
    if os.path.exists(archivo):#Verifica si el archivo existe
        ifile = open(archivo, 'r')     # abre el archivo de entrada
        file_lines = ifile.readlines()  # lee las lineas del archivo
        ifile.close();                  # cierra el archivo de entrada
        a, b, N, Ta, Tb, cf, cb = file_lines[0].split() # separa las columnas de la primera linea
        a = float(a); b = float(b); N = int(N); Ta = float(Ta); Tb = float(Tb); cf = float(cf) ; cb = float(cb) # convierte los datos
        L = ( b - a )
        h = L / (N + 1)
        #cb = 1
       # r = -k / (h**2)
        datos = np.array(9)
        #datos = [a,b,N,L,h,A,B, cf, cb]
        datos=[a,b,N,L,h,Ta,Tb,cf,cb]
        print('\n Datos = [a,b,N,L,h,Ta,Tb,f, b calibración]')
      
       # print('\n Datos = [a,b,N,L,h,Ta,Tb,cf,cb]')
        print(datos)
    else:
        print('No existe el archivo')
        datos=[]
    return datos

#Función que lee datos para la CALIBRACION 2 Neumann

def ReadF3(archivo):
    """
    Lee los parámetros desde un archivo de texto para la calibración 2

    Parameters
    ----------
    archivo : Ruta y nombre del archivo que se quiera leer.

    Returns
    -------
    datos: Arreglo con los datos leídos del archivo

    """
    if os.path.exists(archivo):#Verifica si el archivo existe
        ifile = open(archivo, 'r')     # abre el archivo de entrada
        file_lines = ifile.readlines()  # lee las lineas del archivo
        ifile.close();                  # cierra el archivo de entrada
        a, b, N, Ta, Tb, cf, cb = file_lines[0].split() # separa las columnas de la primera linea
        a = float(a); b = float(b); N = int(N); Ta = float(Ta); Tb = float(Tb); cf = float(cf) ; cb = float(cb) # convierte los datos
        L = ( b - a )
        h = L / (N + 1)
        #cb = 1
       # r = -k / (h**2)
        datos = np.array(9)
        #datos = [a,b,N,L,h,A,B, cf, cb]
        datos=[a,b,N,L,h,Ta,Tb,cf,cb]
        print('\n Datos = [a,b,N,L,h,Ta,Tb,f, b calibración]')
      
       # print('\n Datos = [a,b,N,L,h,Ta,Tb,cf,cb]')
        print(datos)
    else:
        print('No existe el archivo')
        datos=[]
    return datos

 #################################################   

### FUNCIÓN QUE CONSTRUYE LA MATRIZ A DE TAMAÑO NXN
def MatrizA(d, diag):
    """
    Construye la Matriz A de tamanio (N+1) x (N+)

    Parameters
    ----------
    diag : el valor de la diagonal principal que depende de la aproximación
    d : Vector de datos del que se usaran las siguientes posiciones.
    d[2]: Numero de nodos. 

    Returns
    -------
    A : Matriz A del problema


    """
    
    N=d[2]    
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
    return A

def MatAC2(d, A):
    N = d[2]
    cerosN = [i*0 for i in range(0, N-1)]
    #MODIFICAMOS la matriz para el caso de Neumann (AN) cuadno Ta = 0
    AN = []
    for i in A[:-1]:
        AN.append(i+[0])  
    AN.append(A[-1]+[1])
    lN = [1,-1]
    AN.append(cerosN + lN)
    
    #MODIFICAMOS LA MATRIZ AN1 PARA EL CASO DE NEUMANN CUANDO Tb = 0
    AN1 = []
    linversaN = [1,-1]
    AN1.append(linversaN + cerosN) 
    for i in A[:-1]:
        AN1.append(i+[0])   
    AN1.append(A[-1]+[1])
       
    if d[5] == 0:
        print('\n La matriz A con condiciones de Neumann es: ')
        print(np.matrix(AN))
    else: 
        print('\n La matriz A con condiciones de Neumann es: ')
        print(np.matrix(AN1))       
    return AN, AN1

def EspaNume(d):
    """
    Hace el espacio Numérico

    Parameters
    ----------
    d : Vector de datos del que se usaran las siguientes posiciones.
    d[0]: Punto inicial a.
    d[1]: Punto final b.
    d[2]: Numero de nodos.

    Returns
    -------
    Espacio Numérico en X

    """
    a = d[0]
    b = d[1]
    N = d[2]    
    x=np.linspace(a,b,N+1)
    return x

def EspaAna(d):
    """
    Hace el espacio Analítico con 100 muestras

    Parameters
    ----------
    d : Vector de datos del que se usaran las siguientes posiciones.
    d[0]: Punto inicial a.
    d[1]: Punto final b.

    Returns
    -------
    Espacio Numérico en X

    """
    a = d[0]
    b = d[1]    
    #S = d[2]
    x=np.linspace(a,b,100)
#    X2 = [i for i in range(a,b)]
    return x

def SolAna1(x,d):    
    """
    Funcion que regresa la solución analítica del problema 1. 
    Parameters
    ----------
    x: Espacio generado para evaluar la función
    
    d : Vector de datos del que se usaran las siguientes posiciones.
    En el problema 1:
    d[3]: Longitud del segmento.
    d[5]: Condición Dirichlet en A.
    d[6]: Condición Dirichlet en B.
    d[7]: Valor de k.
    d[8]: Valor de q(Escalar). 

    Returns
    -------
    La solución analítica

    """
    #Condicional para selección del problema
    return ((d[6] - d[5])/d[3] + d[8] /(2*d[7]) * (d[3] - x) ) * x + d[5]



def SolAna2(x,d):
    """
    Aun tengo que ajustar aqui los parametros de entrada
    Funcion que regresa la solución analítica de la calibración 1. 
    Parameters
    ----------
    x: Espacio generado para evaluar la función
    
    d : Vector de datos del que se usaran las siguientes posiciones.

    d[7]: Valor de f
    d[8]: Valor de b


    Returns
    -------
    La solución analítica

    """
    f = float(d[7])
    b = float(d[8])
    return ((1-np.cos(f))/(np.sin(f))*np.sin(f*x)+b*np.cos(f*x))

def SolAna3(x):
    """    
    Funcion que regresa la solución analítica del la calibración 2. 
    Parameters
    ----------
    x: Espacio generado para evaluar la función
     
    Returns
    -------
    La solución analítica

    """
    return np.exp(x) - x - np.exp(1) + 4

### MATRIZ QUE LLENA DE LOS DATOS DE Q

def MatQ(d):
    """
    Parameters
    ----------
    d : Vector de datos del que se usaran las siguientes posiciones.
    d[2]: Numero de nodos. 
    d[8]: Valor de q(Escalar). 
    d[9]: Valor de -k / (h**2).
    Returns
    -------
    Devuelve la matriz q

    """
    N=d[2]
    q = np.zeros(N)
    q[1:N-1]=d[8]/d[9]
    print('\n La matriz para Q es:') 
    print(np.array(q))
    return q

#################################################   
### MATRIZ Q PARA CALIBRACIÓN 1

def MatQC1(d):
    """
    d : Vector de datos del que se usaran las siguientes posiciones.
    d[2] = Número de nodos
    
    Parameters
    ----------
 
    
    Returns
    -------
    Devuelve la matriz q

    """
    N=d[2]
    q = np.zeros(N)
    #q[1:N-1]=d[8]/d[9]
    print('\n La matriz para Q es:') 
    print(np.array(q))
    return q

#################################################   
### MATRIZ Q PARA CALIBRACIÓN 2

def MatQC2(d):
    """
    d : Vector de datos del que se usaran las siguientes posiciones.
    d[2] = Número de nodos
    
    Parameters
    ----------
 
    
    Returns
    -------
    Devuelve la matriz q
    """ 
    N = d[2]
    q = np.zeros(N+1)
    #q[-1] = q[-1]/2
    print('\n La matriz para Q es:') 
    print(np.array(q))
    return q

### MATRIZ DE CONDICIONES DE FRONTERA 1

def MatDirichlet(d):
    """
    Hace la matriz para condiciones de Dirichlet
    Parameters
    ----------
    d : Vector de datos del que se usaran las siguientes posiciones.
    d[2]: Numero de nodos N. 
    d[5]: Condición Dirichlet en A.
    d[6]: Condición Dirichlet en B.

    Returns
    -------
    Devuelve la matriz f

    """
    N=d[2]
    f = np.zeros(N)
    f[0] = d[5]
    f[N-1] = d[6]
    print('\n La matriz para los valores de condiciones de frontera es:')
    print(np.array(f))
    
    return f

#MATRIZN CON PROBLEMAS DE FRONTERA TIPO NEUMANN I
def MatNeumman(d):
    """
    Hace la matriz para condiciones de Neumann
    Parameters
    ----------
    N : Tamaño
    Ta : Condición de frontera de Neumann en a
    Tb : Condición de frontera de Neumann en b
    d : lista con todos los valores

    Returns
    -------
    Devuelve la matriz f

    """
    N = d[2]
    r = 1 / (d[4]*d[4])
    #f = np.zeros(N+1)
    f= [] 
    for i in range(1, N+2):
        f.append(np.exp(i*d[4]))
    f[0] *= 0.5
    for i, val in enumerate(f):
        f[i] = val / r
    #print(np.array(f))
    f[-1] = -d[6]
    #N-2 si no funciona N
    
    f[0] = -(d[4] * d[5])
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

def Matb2(Q,f):
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
    bmat = Q + f
    print('\n  matriz b es:')
    print(np.array(bmat))
    return bmat
    
### Definiendo Matriz solución
def sol(A,b,d):
    """
    Soluciona el sistema de ecuaciones
    Parameters
    ----------
    A : Matriz A
    b : Matriz b
    d : Vector de datos del que se usaran las siguientes posiciones.
    d[2]: Numero de nodos. 

    Returns
    -------
    u : Matriz solución, sin considerar las temperaturas en las fronteras

    """
   # print(A, 'A', len(A))
    #print(b, 'B', len(b))

    N=d[2]
    u = np.linalg.solve(A,b)
    # u[0] = Ta
    #u[N+1]=Tb
    print('\n La matriz solución sin condiciones de frontera es:')
    print(np.array(u))
    return u

## Definiendo nueva U para graficar

def u2(u,d):
    """
    Parameters
    ----------
    u : Matriz solución sin considerar temperatura en las fronteras
    Ta : Temperatura en la frontera a
    Tb : Temperatura en la frontera b
    N : Número de Nodos
    d : Vector de datos del que se usaran las siguientes posiciones.
    d[2]: Numero de nodos.
    d[5]: Condición Dirichlet en A.
    d[6]: Condición Dirichlet en B.

    Returns
    -------
    u2 : Matriz solución con temperatura en las fronteras

    """
    #N=d[2]  
    u2 = u
    #u2[0] = d[5]
    #u2[-1] = d[6]
    print('\n La matriz solución con condiciones de frontera es:')

    print(np.array(u2))
    return u2

### Definiendo Matriz solución para Neumann
def solN(B, b, d):
    """
    Soluciona el sistema de ecuaciones
    Parameters
    ----------
    B : Matriz A para el caso de Neumann
    b : Matriz b
    N : Número de nodos

    Returns
    -------
    u : Matriz solución, sin considerar las temperaturas en las fronteras

    """
    u = np.linalg.solve(b, B)
    #u[0] = d[5]
    #u[-1]= d[6]
    print('\n La matriz solución sin condiciones de frontera es:')
    print(np.array(u))
    return u
    

#def Evalua(x,f):
#    u_orig = ((1-np.cos(f))/(np.sin(f))) * np.sin(f*x) + b * np.cos(f*x)
#    return

def Error(d,sa,U):
    """
    Calcula el error, siempre y cuando las dimensiones coincidan.  

    Parameters
    ----------
    d : Vector de datos del que se usaran las siguientes posiciones.
    d[4]: Tamanio de la malla h.

    sa : Solución Analítica.
    U : Solución Numérica.

    Returns
    -------
    None.

    """
   # print('longitud sa', len(sa))
    #print('longitud U', len(U))
    Error = np.sqrt(d[4]) * np.linalg.norm(sa - U)
    print(" Error = %12.10g " % Error)

# PAra graficar

def GrafSol(x,U,xs,sa):
    """
    Genera la grafica que compara solucion analítica a numérica

    Parameters
    ----------
    x : Espacion Numérico.
    U : Solución Numérica.
    xs : Espacio Analítico.
    sa : Solución Analítica. 

    Returns
    -------
    None.

    """
    plt.plot(x,U,'-or', label='Solucion numérica')
    plt.plot(xs,sa,'-b',label='Solucion analítica')
#plt.rcParams["figure.figsize"] = (5, 1)
    plt.xlabel('Dominio')
    plt.ylabel('Temperatura')
    plt.title('Solucion')
    plt.legend(loc='upper left')
#plt.set_cmap('hot')
    plt.show()

def NumNodos(d):
    """
    Función que construye un arreglo que contiene la lista de los nodos 
    Parameters
    ----------
    d : Vector de datos del que se usaran las siguientes posiciones.
    d[2]: Numero de nodos.

    Returns
    -------
    n : arreglo en el qe se guarda de 0 a nodos + 2 (contenplando los extremos)

    """
    tam = d[2] + 2
    n = []
    for i in range(tam): 
        n.append(i)
    return n

def writedat(filename, N, U, Nprecision=2, Uprecision=5):
    """
    Función que genera el archivo de salida
    Parameters
    ----------
    filename : Nombre del archivo de salida.
    N : Arreglo con la lista de los nodos
    U : Solución numérico
    Nprecision : Precisión de los datos para N
    Uprecision : Precisión de los datos para U

    Returns
    -------
    Archivo de salida

    """
    with open(filename,'w') as f:
        for a, b in zip(N, U):
            print("%.*g\t%.*g" % (Nprecision, a, Uprecision, b), file=f)
