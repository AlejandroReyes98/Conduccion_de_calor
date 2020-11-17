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

        nasal=str(input('Nombre del archivo de salida: '))
        datos=f.ReadF3(na)
        if datos:#Verifica si el archivo tiene datos
            opc=5
        else:
            opc=1
    elif opc == 2:
        nasal=str(input('Nombre del archivo de salida: '))
        datos = f.PrePar3()
        opc=5
    elif opc == 3:
        f.Mostrar()
    else:
        print('Opción no válida')
        exit()    
# Se define nuestro sistema Lineal: Matriz A
A, AN1 = f.MatAC2(datos, -2)
x = f.EspaNume1(datos)

#Calibración 2
#Matriz Q
MatrizQ=f.MatQC2(datos)       # Lado derecho
#Matriz f 
MatCondiciones = f.MatNeumman(datos)
#Matriz b)
Matb = MatCondiciones

#Parte que soluciona la matriz 
if datos[5] == 0.0:
    solucion = f.solC2(A,Matb,datos)
else:
    solucion = f.solC2(AN1,Matb,datos)

U = f.u2C2(solucion,datos)

#Solucion analítica:
datos[2] = datos[2]+1
xs=f.EspaAna(datos)
sa=f.SolAna3(xs)
f.Error(datos,f.SolAna3(x),U)

#GRAFICA
f.GrafSol(x,U,xs,sa)  

#Creacion de lista de nodos
Nodos = f.NumNodos(datos[2])

#CREA ARCHIVO

#SE AGREGÓ AL WHILE EL NOMBRE DEL ARCHIVO DE SALIDA
f.writedat(nasal, Nodos, U)