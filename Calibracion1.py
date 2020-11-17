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
        datos=f.ReadF2(na)
        if datos:#Verifica si el archivo tiene datos
            opc=5
        else:
            opc=1
    elif opc == 2:
        nasal=str(input('Nombre del archivo de salida: '))
        datos = f.PrePar2()
        opc=5
    elif opc == 3:
        f.Mostrar()
    else:
        print('Opción no válida')
        exit()     

hc1 = datos[4]
fc1 = datos[7]
diagonal = (hc1**2 * fc1**2) - 2

A = f.MatrizA(datos, diagonal)
x= f.EspaNume(datos)

#Problema 1
#Matriz Q

MatrizQ = f.MatQC1(datos)       # Lado derecho
#Matriz f 
MatCondiciones = f.MatDirichlet(datos)
#Matriz b
Matb = f.Matb(MatrizQ,MatCondiciones)

#Parte que soluciona la matriz 
solucion = f.sol(A,Matb,datos)
U = f.u2(solucion,datos)

#Solucion analítica:
xs=f.EspaAna(datos)
sa=f.SolAna2(xs,datos)

#Error
f.Error(datos,f.SolAna2(x,datos),U)

#GRAFICA
print(x,U,xs,sa)
print(len(x),len(U),len(xs),len(sa))
f.GrafSol(x,U,xs,sa) 

#Creacion de lista de nodos
Nodos = f.NumNodos(datos[2])

#CREA ARCHIVO


#SE AGREGÓ AL WHILE EL NOMBRE DEL ARCHIVO DE SALIDA
f.writedat(nasal, Nodos, U)

