import numpy as np
import scipy.io
import scipy.signal
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

# Filtrado Wiener:

print("Condicionamiento")

A = np.array([[2, 4, 5], [6, 9, 8], [4, 5, 3]])
b = np.array([[220], [490], [274]])

print(A)
print(b)

# vamos a hacerlo de una con la libreria (El numero de condicionamiento):
cond_A = np.linalg.cond(A)
print("El numero de condicionamiento de A es: " + str(cond_A))

# LA ECUACION QUE ESTOS PENSANDO ES:
# A * x = b ===> ¿Se podra?

# Veamos algunas caracteristicas de A

det_A = np.linalg.det(A)
print("Determinante de A:" + str(det_A))

# Lo podemos calcular a mano:

U, S, V = np.linalg.svd(A)
c1 = max(S) / min(np.abs(S))
print("Numero de condicionamiento a mano: " + str(c1))

# podemos hacerlo con los autovalores:

avals_A = np.linalg.eigvals(A)
print("Autovalores: ", avals_A)
c2 = max(avals_A) / min(np.abs(avals_A))
print("El numero de condicionamiento por autovalores: ", c2)
# Esto ultimo fue una aproximacion,

# Calculamos la inversa:

A_inv = np.linalg.inv(A)
x = A_inv.dot(b)
print(x)  # rresultado


# El condicionamiento indica como esta distribuida la energia
# en un sistema

# En el analisis vamos a despejar "b" y asegurar que R este condicionada

# La ecuacion tiene esta forma => R * b = r_xy

# Nos tiene que quedar: b = inv(r)*rxy

def wiener_hopf(x, d, maxlags):  # x:entrada  d:deseada  maxlags:cantidad de retrasos
    rxx = plt.acorr(x, maxlags=maxlags)[1]  # nos vamos a quedar con el segundo parametro
    rxx = rxx[maxlags:-1]  # nos vamos a quedar con el final
    R = toeplitz(rxx)

    rxy = plt.xcorr(x, d, maxlags=maxlags)[1]  # vector con las correlaciones cruzadas
    rxy = rxy[maxlags:-1]

    R_inv = scipy.linalg.inv(R)
    b = R_inv.dot(rxy)

    return b


def sig_noise(f_in, SNR, N):
    fs = 1000
    Ts = 1 / fs

    time = np.arange(0, N) * Ts
    noise = np.random.randn(1, N)

    RMS_noise = np.sqrt(np.mean(noise ** 2))
    t = f_in * np.arange(0, N) * 2 * np.pi / fs
    SNR_n = 10 ** (SNR / 20)
    A = SNR_n * RMS_noise + 1.414
    x = A * np.sin(t)
    x_noise = x + noise

    return x_noise[0], time, x


t_end = 2
fs = 1000
N = t_end * fs
L = 256
SNR = -8
frec = 2

x_noise, t, x = sig_noise(frec, SNR, N)

plt.plot(t, x_noise)
plt.plot(t, x)
plt.show()

b = wiener_hopf(x_noise, x, L)  # Max lags es el orden del filtro
plt.show()

filtrada = scipy.signal.lfilter(b, 1, x_noise)
plt.plot(t, filtrada)
plt.plot(t, x)
plt.show()
plt.xlabel("Time [s]")

print("\n")
print("Filtro Adaptativo!!")


# Tiene un feedback que ayuda a mejorar la salida
# tenemos que usar el gradiente para encontrar el
# camino mas cercano a cero!

# Los coeficientes del filtro van a cambiar y se actualizan
# en forma recursiva con un Delta (Saltos) ===> que vamos a buscar
# para que los pesos sean optimos

# CONFIGURACIONES

# ANC ==> CANCELAR RUIDO (EEG y oculograma) Tengo que aprender cual es el ruido y de esta forma eliminarlo

# ALE ==> Disminuir el ruido en ciertas frecuencias (Como el ruido de linea)

# Esto es tode con analisis estadistico

# INPUT = señal con ruido ==> x[n] + noise[n]
# Canal de referencia = Ruido[n] ==> mediciones en otras partes del cuerpo (ojos, corazon, etc)

# Porque no puedo restar el input al canal de referencia?
# Lo que pasa es que lo que esta en el canal de referencia no es el ruido total

# lo que hace el filtro es aprender la relacion que existe entre las señales

# Armanos la funcion:

# Atencion al DELTA, porque vamos tener que buscar este valor para tener el mejor resultado posible.

def lms(entrada, deseada, L,delta):  # L ==> orden, ir modificando en valor de Delta para obtener el mejor valor posible

    # ERROR CUADRATICO MEDICO

    # La salida es:
    # y = b * x
    # e = deseado - y
    # b = b[-1] + delta * e * x

    # LMS significa error cuadratico medio
    # least mean squares
    # Quiero eliminar el ruido lPM
    M = entrada.shape[0]
    b = np.zeros([1, L])
    y = np.zeros([1, M])
    e = np.zeros([1, M])

    for i in range(L, M):  # tengo que recorrer la señal
        x1 = entrada[i: i - L: -1]  # empiezo desde atras
        # al principio va a ser como la comun y despues va a ir mejorando
        y[0, i] = b.dot(x1)
        e[0, i] = deseada[i] - y[0, i]
        b = b + delta * e[0, i] * x1
    return b  # en este caso con que devuelva b es suficiente porque estamos buscando la optimizacion


t_end = 5
fs = 1000
N = t_end * fs
L = 256
SNR = -8

x_noise, time, x = sig_noise(2, SNR, N)

# LMS     ====>   a las 5 de la tarde!

# 0 < delta < (1/10*L*px)
a = 0.05  # que tanto pesa lo que aprendi para hacer cada iteracion ! ==> Vamos modificando
px = np.mean(x_noise ** 2)
delta = a * (1 / (10 * L * px))

b = lms(x_noise, x, L, delta)

filtrada = scipy.signal.filtfilt(b[0], 1, x_noise)  # filfilt saca el desfazaje

plt.plot(time, filtrada)
plt.plot(time, x)
plt.show()
plt.xlabel("Tiempo [s]")

