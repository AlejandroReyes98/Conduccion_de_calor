"""
Autor: Equipo 1 (azul)  
Fuentes Rubio Natalia Denise - 415119296
Hernández Sandoval Kelly Pamela - 312297473
Reyes Romero Alejandro - 417083191
"""

# Programa principal para el primer problema
# Se llaman las funciones necesarias
import funcionesF as f
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
        datos=f.ReadF1(na)
        nasal=str(input('Nombre del archivo de salida: '))
        if datos:#Verifica si el archivo tiene datos
            opc=5
        else:
            opc=1
    elif opc == 2:
        nasal=str(input('Nombre del archivo de salida: '))
        datos = f.PrePar1()
        opc=5
    elif opc == 3:
        f.Mostrar()
    else:
        print('Opción no válida')
        exit()    
# Se define nuestro sistema Lineal: Matriz A

A = f.MatrizA(datos,-2)
x= f.EspaNume(datos)

#Problema 1
#Matriz Q
MatrizQ=f.MatQ(datos)       # Lado derecho
#Matriz f 
MatCondiciones = f.MatDirichlet(datos)
#Matriz b
Matb = f.Matb(MatrizQ,MatCondiciones)

#Parte que soluciona la matriz 
solucion = f.sol(A,Matb,datos)
U = f.u2(solucion,datos)

#Solucion analítica:
xs=f.EspaAna(datos)
sa=f.SolAna1(xs,datos)

#Error
f.Error(datos,f.SolAna1(x,datos),U)

#GRAFICA
f.GrafSol(x,U,xs,sa)

#Creacion de lista de nodos
Nodos = f.NumNodos(datos[2])

#CREA ARCHIVO

#SE AGREGÓ AL WHILE EL NOMBRE DEL ARCHIVO DE SALIDA
f.writedat(nasal, Nodos, U)
