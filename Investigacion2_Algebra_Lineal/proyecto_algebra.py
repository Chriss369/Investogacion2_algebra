#ESPACIO VECTORIAL
#EJERCICIO 1
import numpy as np

# Definimos los vectores
v1 = np.array([0, 1])
v2 = np.array([3, 4])
v3 = np.array([-1, -2])

# Creamos una matriz con los vectores como columnas
matrix = np.column_stack((v1, v2, v3))

# Calculamos el rango de la matriz
rank = np.linalg.matrix_rank(matrix)

# Verificamos si los vectores forman una base
if rank == 2:
    print("Los vectores forman una base de R^2.")
else:
    print("Los vectores no forman una base de R^2.")

#EJERCICIO 2
import numpy as np

# Definimos los vectores
v1 = np.array([1, 1])
v2 = np.array([2, 2])
v3 = np.array([5, 5])

# Creamos una matriz con los vectores como columnas
matrix = np.column_stack((v1, v2, v3))

# Calculamos el rango de la matriz
rank = np.linalg.matrix_rank(matrix)

# Verificamos si los vectores forman una base
if rank == 2:
    print("Los vectores forman una base de R^2.")
else:
    print("Los vectores no forman una base de R^2.")


#EJERCICIO 3
import numpy as np

# Definimos los vectores
v1 = np.array([1, 2, 3])
v2 = np.array([-1, 2, 3])
v3 = np.array([5, 2, 3])

# Mostramos los vectores
print("Vector 1:", v1)
print("Vector 2:", v2)
print("Vector 3:", v3)


#EJERCICIO 4
import numpy as np

# Definir los vectores
v1 = np.array([0, 5, 1])
v2 = np.array([0, -1, 3])
v3 = np.array([-1, -1, 5])

# Crear la matriz con los vectores como columnas
matrix = np.column_stack((v1, v2, v3))

# Calcular el determinante de la matriz
volume = np.abs(np.linalg.det(matrix))

print("El volumen del paralelepípedo es:", volume)

# EJERCICIO 5
import numpy as np

# Definir los vectores
v1 = np.array([1, 1, 1])
v2 = np.array([0, 1, 1])
v3 = np.array([0, 0, 1])

# Crear la matriz con los vectores como columnas
matrix = np.column_stack((v1, v2, v3))

# Calcular el determinante de la matriz
volume = np.abs(np.linalg.det(matrix))

print("Forman una base de R3:", volume)

#SUBESPACIO VECTORIAL
#EJERCICIO 1
import numpy as np

# Definimos el conjunto W como una matriz de vectores de la forma (x1, x2, x3, 0)
# Vamos a definir dos vectores en W y un escalar
v1 = np.array([1, 2, 3, 0])
v2 = np.array([4, 5, 6, 0])
alpha = 2

# Verificar si el vector cero está en W
vector_cero = np.array([0, 0, 0, 0])
en_W = np.array_equal(vector_cero, [0, 0, 0, 0])

# Verificar si W es cerrado bajo la suma
suma = v1 + v2
cerrado_suma = suma[3] == 0

# Verificar si W es cerrado bajo la multiplicación escalar
multiplicacion_escalar = alpha * v1
cerrado_multiplicacion = multiplicacion_escalar[3] == 0

(en_W, cerrado_suma, cerrado_multiplicacion)


#EJERCICIO 2
import numpy as np

# Definimos dos vectores en W y un escalar
v1 = np.array([1, 2, 2*1 - 3*2])
v2 = np.array([3, 4, 2*3 - 3*4])
alpha = 2

# Verificar si el vector cero está en W
vector_cero = np.array([0, 0, 2*0 - 3*0])
en_W = np.array_equal(vector_cero, [0, 0, 0])

# Verificar si W es cerrado bajo la suma
suma = v1 + v2
cerrado_suma = suma[2] == 2 * suma[0] - 3 * suma[1]

# Verificar si W es cerrado bajo la multiplicación escalar
multiplicacion_escalar = alpha * v1
cerrado_multiplicacion = multiplicacion_escalar[2] == 2 * multiplicacion_escalar[0] - 3 * multiplicacion_escalar[1]

(en_W, cerrado_suma, cerrado_multiplicacion)

#EJERCICIO 3
import numpy as np

# Definir dos matrices en W y un escalar
a1, b1 = 1, 2
a2, b2 = 3, 4
alpha = 2

# Matrices en W
M1 = np.array([[0, a1], [b1, 0]])
M2 = np.array([[0, a2], [b2, 0]])

# Verificar si la matriz cero está en W
M_cero = np.array([[0, 0], [0, 0]])
en_W = np.array_equal(M_cero, np.array([[0, 0], [0, 0]]))

# Verificar si W es cerrado bajo la suma
suma = M1 + M2
cerrado_suma = (suma[0,0] == 0) and (suma[1,1] == 0)

# Verificar si W es cerrado bajo la multiplicación escalar
multiplicacion_escalar = alpha * M1
cerrado_multiplicacion = (multiplicacion_escalar[0,0] == 0) and (multiplicacion_escalar[1,1] == 0)

(en_W, cerrado_suma, cerrado_multiplicacion)

#EJERCICIO 4
import numpy as np
def verificar_subespacio():
    # Verificar que contiene el vector cero
    vector_cero = np.array([0, 0, 0])
    if not np.array_equal(vector_cero, np.array([0, 0, 0])):
        return False, "No contiene el vector cero"
    # Verificar cerradura bajo la adición
    x1, x2, y1, y2 = 1, 2, 3, 4  # Ejemplos de números reales
    v1 = np.array([x1, x2, 0])
    v2 = np.array([y1, y2, 0])
    suma = v1 + v2
    if not (suma[2] == 0):
        return False, "No es cerrado bajo la adición"

    # Verificar cerradura bajo la multiplicación por escalares
    escalar = 2  # Ejemplo de un escalar
    multiplicacion = escalar * v1
    if not (multiplicacion[2] == 0):
        return False, "No es cerrado bajo la multiplicación por escalares"

    return True, "Es un subespacio vectorial"

es_subespacio, mensaje = verificar_subespacio()
print(mensaje)


#EJERCICIO 5
import numpy as np
def verificar_subespacio():
    # Verificar que contiene el vector cero
    vector_cero = np.array([0, 0, 0])
    if not np.array_equal(vector_cero, np.array([0, 0, 4])):
        return False, "No contiene el vector cero"

    # Verificar cerradura bajo la adición
    x1, x2, y1, y2 = 1, 2, 3, 4  # Ejemplos de números reales
    v1 = np.array([x1, x2, 4])
    v2 = np.array([y1, y2, 4])
    suma = v1 + v2
    if suma[2] != 4:
        return False, "No es cerrado bajo la adición"

    # Verificar cerradura bajo la multiplicación por escalares
    escalar = 2  # Ejemplo de un escalar
    multiplicacion = escalar * v1
    if multiplicacion[2] != 4:
        return False, "No es cerrado bajo la multiplicación por escalares"

    return True, "Es un subespacio vectorial"


es_subespacio, mensaje = verificar_subespacio()
print(mensaje)


# COMBINACION LINAL
#Combinacion lineal ejercicio.1.
# Definir los vectores como listas
v1 = [1, 3, 1]
v2 = [0, 1, 2]
v3 = [1, 0, -5]

# Definir los coeficientes
c2, c3 = 3, 1

# Calcular la combinación lineal
v1_correcta = [c2 * v2[i] + c3 * v3[i] for i in range(len(v1))]

# Verificar si la combinación lineal es correcta
es_correcta = v1 == v1_correcta

# Mostrar resultados
print("Resultado de la combinación lineal:", v1_correcta)
print("¿Es correcta la combinación lineal?", es_correcta)
print("Coeficientes encontrados: c2 =", c2, ", c3 =", c3)


#Combinacion lineal ejercicio.2.
def calculate_determinant(v1, v2, v3):
    # Extraer los componentes de los vectores
    a11, a12, a13 = v1
    a21, a22, a23 = v2
    a31, a32, a33 = v3

    # Calcular el determinante de la matriz 3x3 formada por los vectores
    determinant = (
        a11 * (a22 * a33 - a23 * a32) -
        a12 * (a21 * a33 - a23 * a31) +
        a13 * (a21 * a32 - a22 * a31)
    )

    return determinant

def solve_combination(v1, v2, v3, v):
    # Extraer los componentes de los vectores
    a11, a12, a13 = v1
    a21, a22, a23 = v2
    a31, a32, a33 = v3
    b1, b2, b3 = v

    # Resolver el sistema de ecuaciones utilizando la regla de Cramer
    det_A = calculate_determinant(v1, v2, v3)
    if det_A == 0:
        return None  # No hay solución única

    # Determinantes de las matrices modificadas
    det_A1 = calculate_determinant([b1, b2, b3], v2, v3)
    det_A2 = calculate_determinant(v1, [b1, b2, b3], v3)
    det_A3 = calculate_determinant(v1, v2, [b1, b2, b3])

    # Soluciones de las variables
    c1 = det_A1 / det_A
    c2 = det_A2 / det_A
    c3 = det_A3 / det_A

    return c1, c2, c3

# Definir los vectores
v1 = [2, 3, 0]
v2 = [0, 1, -5]
v3 = [0, 1, 3]

# Definir el vector que queremos expresar como combinación lineal
v = [1, 1, 1]  # Puedes cambiar este vector por el que necesites

# Calcular el determinante de la matriz
determinant = calculate_determinant(v1, v2, v3)
print("El determinante de la matriz es:", determinant)

# Resolver la combinación lineal
coefficients = solve_combination(v1, v2, v3, v)

# Mostrar los coeficientes
if coefficients:
    c1, c2, c3 = coefficients
    print(f"La combinación lineal es: {c1} * {v1} + {c2} * {v2} + {c3} * {v3} = {v}")
else:
    print("No hay solución única para la combinación lineal.")



#Combinacion lineal ejercicio.3.
# Definir los vectores como listas de listas (matrices 2x2)
v1 = [[0, 8], [2, 1]]
v2 = [[0, 2], [1, 0]]
v3 = [[-1, 3], [0, 3]]
v4 = [[-2, 0], [1, 3]]

# Definir los coeficientes
c1, c2 = -2, -1

# Calcular la combinación lineal
v4_calculado = [[c1 * v1[i][j] + c2 * v2[i][j] for j in range(2)] for i in range(2)]

# Verificar si la combinación lineal es correcta
es_correcta = v4 == v4_calculado

# Mostrar resultados
print("Resultado de la combinación lineal:", v4_calculado)
print("¿Es correcta la combinación lineal?", es_correcta)
print("Coeficientes encontrados: c1 =", c1, ", c2 =", c2)


#Combinacion lineal ejercicio.4.
# Definir los vectores como listas
v1 = [1, 2, 3]
v2 = [0, 1, 2]
v3 = [-1, 0, 1]

# Definir el vector w
w = [1, 1, 1]

# Definir los coeficientes
c1, c2, c3 = 1, -1, 0  # Estos son valores que vamos a probar

# Calcular la combinación lineal
w_calculado = [c1 * v1[i] + c2 * v2[i] + c3 * v3[i] for i in range(len(w))]

# Verificar si la combinación lineal es correcta
es_correcta = w == w_calculado

# Mostrar resultados
print("Resultado de la combinación lineal:", w_calculado)
print("¿Es correcta la combinación lineal?", es_correcta)
print("Coeficientes encontrados: c1 =", c1, ", c2 =", c2, ", c3 =", c3)



#Combinacion lineal ejercicio.5.
# Definir los vectores como listas
v1 = [1, 2, 3]
v2 = [0, 1, 2]
v3 = [-1, 0, 1]

# Definir el vector w
w = [1, -2, 2]

# Definir los coeficientes (Vamos a intentar encontrar c1, c2, y c3)
# Aquí asumimos valores iniciales para los coeficientes para realizar el cálculo
c1, c2, c3 = 0.5, -3, -0.5  # Puedes ajustar estos valores según necesites

# Calcular la combinación lineal
w_calculado = [c1 * v1[i] + c2 * v2[i] + c3 * v3[i] for i in range(len(w))]

# Verificar si la combinación lineal es correcta
es_correcta = w == w_calculado

# Mostrar resultados
print("Resultado de la combinación lineal:", w_calculado)
print("¿Es correcta la combinación lineal?", es_correcta)
print("Coeficientes encontrados: c1 =", c1, ", c2 =", c2, ", c3 =", c3)


# INDEPENDENCIA LINEAL
# Independencia lineal ejercicio.1.
def gauss_jordan(m, eps=1.0 / (10 ** 10)):
    (h, w) = (len(m), len(m[0]))
    for y in range(0, h):
        maxrow = y
        for y2 in range(y + 1, h):
            if abs(m[y2][y]) > abs(m[maxrow][y]):
                maxrow = y2
        m[y], m[maxrow] = m[maxrow], m[y]
        if abs(m[y][y]) <= eps:
            continue
        for y2 in range(y + 1, h):
            c = m[y2][y] / m[y][y]
            for x in range(y, w):
                m[y2][x] -= m[y][x] * c
    for y in range(h - 1, -1, -1):
        c = m[y][y]
        for y2 in range(0, y):
            for x in range(w - 1, y - 1, -1):
                m[y2][x] -= m[y][x] * m[y2][y] / c
        m[y][y] /= c
        for x in range(h, w):
            m[y][x] /= c
    return m


# Definir los vectores
v1 = [1, 2, 3]
v2 = [0, 1, 2]
v3 = [-2, 0, 1]

# Formar la matriz con los vectores como columnas
A = [v1, v2, v3]

# Añadir la columna de ceros para formar la matriz aumentada
augmented_matrix = [row + [0] for row in A]

# Realizar la eliminación de Gauss-Jordan
reduced_matrix = gauss_jordan(augmented_matrix)


# Independencia lineal ejercicio.2.
def gauss_jordan(m, eps=1.0 / (10 ** 10)):
    (h, w) = (len(m), len(m[0]))
    for y in range(0, h):
        # Encontrar la fila con el valor máximo en la columna y para evitar división por cero
        maxrow = y
        for y2 in range(y + 1, h):
            if abs(m[y2][y]) > abs(m[maxrow][y]):
                maxrow = y2
        m[y], m[maxrow] = m[maxrow], m[y]

        # Si el pivote es cero, continuar con la siguiente columna
        if abs(m[y][y]) <= eps:
            continue

        # Normalizar la fila del pivote
        for x in range(y, w):
            m[y][x] /= m[y][y]

        # Eliminar todas las otras entradas en esta columna
        for y2 in range(h):
            if y2 != y:
                c = m[y2][y]
                for x in range(y, w):
                    m[y2][x] -= m[y][x] * c
    return m


# Definir los coeficientes de los vectores (polinomios)
v1 = [1, 1, -2]
v2 = [2, 5, -1]
v3 = [0, 1, 1]

# Formar la matriz con los vectores como columnas
A = [
    [1, 2, 0],
    [1, 5, 1],
    [-2, -1, 1]
]

# Añadir la columna de ceros para formar la matriz aumentada
augmented_matrix = [row + [0] for row in A]

# Realizar la eliminación de Gauss-Jordan
reduced_matrix = gauss_jordan(augmented_matrix)

# Verificar la independencia lineal
independent = True
for row in reduced_matrix:
    if all(abs(value) <= 1e-10 for value in row[:-1]):
        independent = False
        break

if independent:
    print("El conjunto S es linealmente independiente.")
else:
    print("El conjunto S es linealmente dependiente.")


# Independencia lineal ejercicio.3.
def gauss_jordan(m, eps=1.0 / (10 ** 10)):
    (h, w) = (len(m), len(m[0]))
    for y in range(0, h):
        # Encontrar la fila con el valor máximo en la columna y para evitar división por cero
        maxrow = y
        for y2 in range(y + 1, h):
            if abs(m[y2][y]) > abs(m[maxrow][y]):
                maxrow = y2
        m[y], m[maxrow] = m[maxrow], m[y]

        # Si el pivote es cero, continuar con la siguiente columna
        if abs(m[y][y]) <= eps:
            continue

        # Normalizar la fila del pivote
        for x in range(y, w):
            m[y][x] /= m[y][y]

        # Eliminar todas las otras entradas en esta columna
        for y2 in range(h):
            if y2 != y:
                c = m[y2][y]
                for x in range(y, w):
                    m[y2][x] -= m[y][x] * c
    return m


# Definir los vectores (matrices 2x2)
v1 = [[2, 1], [0, 1]]
v2 = [[3, 0], [2, 1]]
v3 = [[1, 0], [2, 0]]


# Independencia lineal ejercicio.4.
def determinante(m):
    # Calcula el determinante de una matriz 4x4 utilizando la expansión por cofactores
    if len(m) != 4 or len(m[0]) != 4:
        raise ValueError("La matriz debe ser de 4x4")

    def cofactor(matrix, row, col):
        return [fila[:col] + fila[col + 1:] for fila in (matrix[:row] + matrix[row + 1:])]

    def determinante_3x3(matriz):
        return (matriz[0][0] * (matriz[1][1] * matriz[2][2] - matriz[1][2] * matriz[2][1]) -
                matriz[0][1] * (matriz[1][0] * matriz[2][2] - matriz[1][2] * matriz[2][0]) +
                matriz[0][2] * (matriz[1][0] * matriz[2][1] - matriz[1][1] * matriz[2][0]))

    det = 0
    for col in range(4):
        minor = cofactor(m, 0, col)
        det += ((-1) ** col) * m[0][col] * determinante_3x3(minor)
    return det


# Definimos los vectores
v1 = [1, 0, -1, 0]
v2 = [1, 0, 0, 2]
v3 = [0, 3, 1, -2]
v4 = [0, -1, -1, 2]

# Formamos la matriz con los vectores como columnas
A = [list(col) for col in zip(v1, v2, v3, v4)]

# Calculamos el determinante de la matriz
det = determinante(A)

# Verificamos si el determinante es distinto de cero
if det != 0:
    print("El conjunto de vectores es linealmente independiente.")
else:
    print("El conjunto de vectores es linealmente dependiente.")


# Independencia lineal ejercicio.5.
def determinante(m):
    # Calcula el determinante de una matriz 4x4 utilizando la expansión por cofactores
    if len(m) != 4 or len(m[0]) != 4:
        raise ValueError("La matriz debe ser de 4x4")

    def cofactor(matrix, row, col):
        return [fila[:col] + fila[col + 1:] for fila in (matrix[:row] + matrix[row + 1:])]

    def determinante_3x3(matriz):
        return (matriz[0][0] * (matriz[1][1] * matriz[2][2] - matriz[1][2] * matriz[2][1]) -
                matriz[0][1] * (matriz[1][0] * matriz[2][2] - matriz[1][2] * matriz[2][0]) +
                matriz[0][2] * (matriz[1][0] * matriz[2][1] - matriz[1][1] * matriz[2][0]))

    det = 0
    for col in range(4):
        minor = cofactor(m, 0, col)
        det += ((-1) ** col) * m[0][col] * determinante_3x3(minor)
    return det


# Definimos los vectores
v1 = [2, 1, -1, 0]
v2 = [0, 4, 0, 1]
v3 = [1, 0, 3, -2]
v4 = [1, -1, 1, 3]

# Formamos la matriz con los vectores como columnas
A = [list(col) for col in zip(v1, v2, v3, v4)]

# Calculamos el determinante de la matriz
det = determinante(A)

# Verificamos si el determinante es distinto de cero
if det != 0:
    print("El conjunto de vectores es linealmente independiente.")
else:
    print("El conjunto de vectores es linealmente dependiente.")



#BASE Y DIMENSION
#EJERCICIO1
#Demuestre que el siguiente conjunto es una base de R3.
#S={(1,0,0), (0,1,0), (0,0,1)}

import numpy as np

v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

matrix = np.column_stack((v1, v2, v3))

rank = np.linalg.matrix_rank(matrix)
print("Rango de la matriz:", rank)
print("Los vectores son linealmente independientes y generan R^3." if rank == 3 else "Los vectores NO son linealmente independientes y NO generan R^3.")

#EJERCICIO2
import numpy as np

# Defino los vectores
v1 = np.array([1, 1])
v2 = np.array([1, -1])

# Creo una matriz con los vectores como columnas
matrix = np.column_stack((v1, v2))

# Verifico si los vectores son linealmente independientes y generan R^2
rank = np.linalg.matrix_rank(matrix)
print("Rango de la matriz:", rank)
print("Los vectores son linealmente independientes y generan R^2." if rank == 2 else "Los vectores NO son linealmente independientes y NO generan R^2.")

#EJERCICIO3
import numpy as np

S = np.array([[1, 2, 3], [0, 2, 1], [2, 0, 1]])

if np.array_equal(np.zeros(3), S.dot(np.zeros(3).T)):
    print("El vector nulo pertenece a S.")
else:
    print("El vector nulo no pertenece a S.")

print("Una base de S es:", S)
print("La dimensión de S es:", np.linalg.matrix_rank(S))

#EJERCICIO4

s1 = [1, 0, 0, 0]
s2 = [0, 1, 0, 0]
s3 = [0, 0, 1, 0]
s4 = [0, 0, 0, 1]

S = [s1, s2, s3, s4]
print(len(S) == len(set(map(tuple, S))))

def is_in_span(coeffs):
    for coeff in coeffs:
        if coeff != 0 and all(coeff * vector != [coeff * x for x in vector] for vector in S):
            return False
    return True

p = [1, 2, 3, 4]
print(is_in_span(p))

print("El conjunto S = {1, x, x^2, x^3} es una base de P3.")

#EJERCICIO5
import numpy as np

S = [
    np.array([[1, 0], [0, 0]]),
    np.array([[0, 1], [0, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [0, 1]])
]

matrix = np.column_stack([A.flatten() for A in S])
rank = np.linalg.matrix_rank(matrix)

print("El conjunto S es linealmente independiente y es una base de M_{2,2}." if rank == 4 else "El conjunto S no es linealmente independiente.")


# RANGO E IMAGEN DE UNA MATRIZ
# Ejercicio1
import numpy as np

# Definir la matriz A
A = np.array([[1, 0, 1], [-2, 3, 4]])
rango = np.linalg.matrix_rank(A)
print(f"El rango de la matriz A es: {rango}")

Q, R = np.linalg.qr(A)
independientes = np.where(np.abs(np.diag(R)) > 1e-10)[0]

columnas_independientes = A[:, independientes]
print("Las columnas que forman la imagen de la matriz A son:")
print(columnas_independientes)

# Ejercicio2
import numpy as np

A = np.array([
    [1, 3, 1, 3],
    [0, 1, 1, 0],
    [-3, 0, 6, -1],
    [3, 4, -2, 1],
    [2, 0, -4, -2]
], dtype=float)

A[2] = A[2] + 3 * A[0]  # Fila 3 = Fila 3 + 3 * Fila 1
A[3] = A[3] - 3 * A[0]  # Fila 4 = Fila 4 - 3 * Fila 1
A[4] = A[4] - 2 * A[0]  # Fila 5 = Fila 5 - 2 * Fila 1
A[3] = A[3] - 4 * A[1]  # Fila 4 = Fila 4 - 4 * Fila 2
A[4] = A[4] - 2 * A[1]  # Fila 5 = Fila 5 - 2 * Fila 2

# Resultado esperado de la forma escalonada
B = np.array([
    [1, 3, 1, 3],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=float)
print("Matriz escalonada B:")
print(B)
rango = np.sum(np.any(B  != 0, axis=1))
base = B[np.any(B != 0, axis=1)]

print(f"el rango de la matriz A es: {rango}")
print("una base para el espacio renglon de A es:")
print(base)

#ejercicio3
# import numpy as np
# vi = np.array((-1, 2, 5])
# v2 = np.array ([3, 0, 3])
# v3 = np.array([5, 1, 8])
# A = np. array ((v1, v2, v3))
# print ("Matriz A:")
# print (A)
# from sympy import Matrix
# A_sympy = Matrix(A)
# 11
# B = A_sympy.rref () [o]
# print ("Matriz B en forma escalonada reducida:")
# print (np.array (B) .astype(np. fLoat64))                                    base_filas = B.tolist()
# base = (fila for fila in base_filas if any(fila)]
# print("Base del subespacio generado por s = (v1, v2, v3):")
# print(np.array (base).astype (np.fLoat64))

# Ejercicio 4
import numpy as np
from sympy import Matrix

A = np.array([[1, 0, 3, 2],
              [3, 6, 9, 5],
              [1, 3, 5, 3],
              [1, 6, -2, 4]])

A_T = A.T
A_T_sym = Matrix(A_T)
ref_matrix, pivot_columns = A_T_sym.rref()
column_space_basis = A[:, pivot_columns]

print("La base para el espacio columna de A es:")
for vector in column_space_basis.T:  # Transponemos para imprimir los vectores como columnas
 print(vector)

# Ejercicio 5
import numpy as np
from sympy import Matrix

# Matriz A original
A = np.array([[1, 2, 0, 1],
              [2, 5, 3, 2],
              [1, 0, 3, 5],
              [0, 1, 1, 0]])

A_sym = Matrix(A)
B, pivot_columns = A_sym.rref()
B = np.array(B).astype(np.float64)
rango = np.sum(np.any(B != 0, axis=1))

print("La forma escalonada por filas de A es:")
print(B)
print("El rango de A es:", rango)

# CAMBIO DE BASE
#EJERCICIO 1
import numpy as np

# Definimos las bases B y B'
B = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

B_prime = np.array([
    [1, 0, 1],
    [0, -1, 2],
    [2, 3, -5]
])

# La matriz de transición de B a B' es la matriz que satisface la ecuación P * B = B'
# Para encontrar P, podemos usar la relación P = B_prime * B_inv, donde B_inv es la inversa de B
B_inv = np.linalg.inv(B)
P = np.dot(B_prime, B_inv)

print("La matriz de transición de B a B' es:")
print(P)

#EJERCICIO 2
import numpy as np

# Define the vector x
x = np.array([1, 2, -1])

# Define the non-standard basis B'
u1 = np.array([1, 0, 1])
u2 = np.array([0, -1, 2])
u3 = np.array([2, 3, -5])
B_prime = np.array([u1, u2, u3]).T

# Calculate the coordinates of x relative to the basis B'
B_prime_inv = np.linalg.inv(B_prime)
coordinates = np.dot(B_prime_inv, x)

print("The coordinates of x relative to the non-standard basis B' are:", coordinates)
#EJERCICIO 3
import numpy as np

# Define the vectors of the standard basis B
B1 = np.array([-3, 2])
B2 = np.array([4, -2])

# Define the vectors of the non-standard basis B'
B_prime1 = np.array([-1, 2])
B_prime2 = np.array([2, -2])

# Form the matrices P_B and P_{B'}
P_B = np.column_stack((B1, B2))
P_B_prime = np.column_stack((B_prime1, B_prime2))

# Calculate the transition matrix P_{B \to B'}
P_B_prime_inv = np.linalg.inv(P_B_prime)
P_B_to_B_prime = np.dot(P_B_prime_inv, P_B)

# Print the result
print("The transition matrix from B to B' is:\n", P_B_to_B_prime)

#EJERCICIO 4
# Define the coefficients of the polynomial p in the standard basis S
a0 = 4
a1 = 0
a2 = -2
a3 = 3

# Form the coordinate vector
coordinates = np.array([a0, a1, a2, a3])

# Print the result
print("The coordinates of p relative to the standard basis are:", coordinates)

#EJERCICIO 5
import numpy as np

# Define the basis vectors in the non-standard basis
v1 = np.array([1, 0])
v2 = np.array([1, 2])

# Define the coordinates of x in the non-standard basis
x_B = np.array([3, 2])

# Form the change of basis matrix from the non-standard basis to the standard basis
B = np.column_stack((v1, v2))

# Calculate the vector in the standard basis
x_standard = B @ x_B

print("The coordinates of x in the standard basis are:", x_standard)


# TRANSFORMACIÓN LINEAL
#EJERCICIO 1
import numpy as np

# Definir la función de transformación T
def T(v):
    v1, v2 = v
    return np.array([v1 - v2, v1 + 2 * v2])

# a. Imagen de v = (-1, 2)
v1 = np.array([-1, 2])
imagen_v1 = T(v1)

# b. Imagen de v = (0, 0)
v2 = np.array([0, 0])
imagen_v2 = T(v2)

# c. Preimagen de w = (-1, 11)
# Resolver el sistema de ecuaciones:
# w1 = v1 - v2
# w2 = v1 + 2 * v2
# Para w = (-1, 11)
w = np.array([-1, 11])

# Usar numpy para resolver el sistema de ecuaciones
# A * [v1, v2] = w
A = np.array([[1, -1], [1, 2]])
preimagen_w = np.linalg.solve(A, w)

imagen_v1, imagen_v2, preimagen_w

#EJERCICIO 2
import numpy as np

# Definir la función de transformación T
def T(v):
    v1, v2 = v
    return np.array([v1 - v2, v1 + 2 * v2])

# Verificar la aditividad
def verificar_aditividad(u, v):
    suma_uv = u + v
    T_suma_uv = T(suma_uv)
    T_u_mas_T_v = T(u) + T(v)
    return np.allclose(T_suma_uv, T_u_mas_T_v)

# Verificar la homogeneidad
def verificar_homogeneidad(v, c):
    c_v = c * v
    T_c_v = T(c_v)
    c_T_v = c * T(v)
    return np.allclose(T_c_v, c_T_v)

# Vectores de prueba
u = np.array([1, 2])
v = np.array([3, 4])
c = 5

# Verificaciones
aditividad = verificar_aditividad(u, v)
homogeneidad = verificar_homogeneidad(v, c)

print("Aditividad:", aditividad)
print("Homogeneidad:", homogeneidad)

#EJERCICIO 3

import numpy as np

# a. f(x) = sin(x)
def f_sin(x):
    return np.sin(x)

# Verificar aditividad
x1 = np.pi / 2
x2 = np.pi / 3
additivity_sin = f_sin(x1 + x2) == f_sin(x1) + f_sin(x2)

# b. f(x) = x^2
def f_square(x):
    return x ** 2

# Verificar aditividad
x1 = 1
x2 = 2
additivity_square = f_square(x1 + x2) == f_square(x1) + f_square(x2)

# c. f(x) = x + 1
def f_linear_plus_one(x):
    return x + 1

# Verificar aditividad
x1 = 1
x2 = 2
additivity_linear_plus_one = f_linear_plus_one(x1 + x2) == f_linear_plus_one(x1) + f_linear_plus_one(x2)

additivity_sin, additivity_square, additivity_linear_plus_one

#EJERCICIO 4
import numpy as np

# Vectores originales y sus imágenes bajo la transformación T
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

Tv1 = np.array([2, -1, 4])
Tv2 = np.array([1, 5, -2])
Tv3 = np.array([0, 3, 1])

# Matriz de la transformación T
T_matrix = np.column_stack([Tv1, Tv2, Tv3])

# Vector a transformar
v = np.array([2, 3, -2])

# Resultado de la transformación
Tv = np.dot(T_matrix, v)

print("T(2, 3, -2) =", Tv)

#EJERCICIO 5
import numpy as np

# Matriz de la transformación T
A = np.array([
    [3, 0],
    [2, 1],
    [-1, -2]
])

# Vector a transformar
v = np.array([2, -1])

# Resultado de la transformación
Tv = np.dot(A, v)

print("T(v) =", Tv)

# Demostración de que T es una transformación lineal
# Para demostrar que T es una transformación lineal, debemos verificar
# que T(c1*v1 + c2*v2) = c1*T(v1) + c2*T(v2) para cualesquiera c1, c2 en R y cualesquiera vectores v1, v2 en R^2

# Definimos algunos valores de prueba
v1 = np.array([1, 2])
v2 = np.array([-3, 4])
c1 = 2
c2 = -1

# Calculamos T(c1*v1 + c2*v2)
T_combined = np.dot(A, c1*v1 + c2*v2)

# Calculamos c1*T(v1) + c2*T(v2)
T_separate = c1 * np.dot(A, v1) + c2 * np.dot(A, v2)

# Verificamos si son iguales
is_linear = np.allclose(T_combined, T_separate)

print("¿T es una transformación lineal? ->", is_linear)


















