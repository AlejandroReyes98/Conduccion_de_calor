# -*- coding: utf-8 -*-
"""
Programa

V E R S I O N   6        12/11/2020
-Se separan las funciones por cada calibración, problema
-Se soluciona el problema 1 sin fuentes y con fuentes(O J O:q en todas partes)
-Se implementa la función que lee de un archivo
-Quiero ver si es posible minimizar las variables en los archivos principales(LISTO)
-Creo que el primer problema con condiciones de Dirichlet ya esta listo
-Solo faltaria escribir en un archivo los resultados
"""
# Programa principal para el primer problema
# Se llaman las funciones necesarias
import funcionesv7c2 as f
#Imprimimos el titulo
f.Titulo()
#Leemos los parametros de entrada:
opc=0
while opc < 4: 
    print('|¿Cuenta con archivo de parametros?                |')
    print('|  1)Si     2)No    3)Revisar                      |')
    print('|--------------------------------------------------|')
    opc=int(input('|Ingrese el número de la opción: '))

    if opc == 1:
        #datos=f.PrePar1()
        na=str(input('|Nombre del archivo: '))
         #################################################   

        nasal=str(input('Nombre del archivo de salida: '))
        datos=f.ReadF3(na)
        if datos:#Verifica si el archivo tiene datos
            opc=5
        else:
            opc=1
    elif opc == 2:
        datos = f.PrePar3()
        opc=5
    elif opc == 3:
        f.Mostrar()
    else:
        print('Opción no válida')
        exit()    
# Se define nuestro sistema Lineal: Matriz A
#MatrizA(N,diag,k)
A = f.MatrizA(datos, -2)
AN, AN1 = f.MatAC2(datos, A)
x = f.EspaNume(datos)

#Calibración 2
#Matriz Q
MatrizQ=f.MatQC2(datos)       # Lado derecho
#Matriz f 
MatCondiciones = f.MatNeumman(datos)
#Matriz b
#Matb = f.Matb(MatrizQ,MatCondiciones)
#print('estos es matcondiciones', MatCondiciones)
Matb = MatCondiciones

#Parte que soluciona la matriz 
if datos[5] == 0.0:
    solucion = f.sol(AN,Matb,datos)
else:
    solucion = f.sol(AN1,Matb,datos)

U = f.u2(solucion,datos)

#Solucion analítica:
datos[2] = datos[2]+1
xs=f.EspaAna(datos)
sa=f.SolAna3(xs)
#print(U,'U')
#print(xs, 'XS')
#print(sa, 'SA')

#f.Error(datos,sa,U)
f.Error(datos,f.SolAna3(x),U)

#GRAFICA
f.GrafSol(x,U,xs,sa)
#f.GrafSol(x,sa)

###############################################   

#Creacion de lista de nodos
Nodos = f.NumNodos(datos)

#CREA ARCHIVO

#SE AGREGÓ AL WHILE EL NOMBRE DEL ARCHIVO DE SALIDA
f.writedat('PARAMETROSC2', Nodos, U)