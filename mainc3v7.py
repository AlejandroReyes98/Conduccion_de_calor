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
import funcionesc3v7 as f

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
        datos=f.ReadF4(na)
        nasal=str(input('Nombre del archivo de salida: '))
        if datos:#Verifica si el archivo tiene datos
            opc=5
        else:
            opc=1
    elif opc == 2:
        datos = f.PrePar4()
        nasal=str(input('Nombre del archivo de salida: '))
        opc=5
    elif opc == 3:
        f.Mostrar()
    else:
        print('Opción no válida')
        exit()    
# Se define nuestro sistema Lineal: Matriz A
#MatrizA(N,diag,k)


x= f.EspaNume(datos)
k=f.TipoK(x)
pk=f.pk(k)
A = f.MatAc3(datos,pk)
#Problema 1
#Matriz Q
MatrizQ=f.MatQc3(datos)       # Lado derecho
#Matriz f 
MatCondiciones = f.MatDirichletc3(datos,pk)
#Matriz b
Matb = f.Matbc3(MatrizQ,MatCondiciones)

#Parte que soluciona la matriz 
solucion = f.sol(A,Matb,datos)
U = f.u2(solucion,datos)

#GRAFICA
f.GrafSolc3(x,k,U)

#Creacion de lista de nodos
Nodos = f.NumNodos(datos)

#CREA ARCHIVO

#SE AGREGÓ AL WHILE EL NOMBRE DEL ARCHIVO DE SALIDA
f.writedat(nasal, Nodos, U)