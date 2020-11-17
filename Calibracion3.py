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
Matb = f.Matb2(MatrizQ,MatCondiciones)

#Parte que soluciona la matriz 
solucion = f.sol(A,Matb,datos)
U = f.u2(solucion,datos)

#GRAFICA
f.GrafSolc3(x,k,U)

#Creacion de lista de nodos
Nodos = f.NumNodos(datos[2])

#CREA ARCHIVO

#SE AGREGÓ AL WHILE EL NOMBRE DEL ARCHIVO DE SALIDA
f.writedat(nasal, Nodos, U)