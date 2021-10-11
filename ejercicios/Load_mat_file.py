import numpy as np
import scipy as sp
from scipy.fft import fft, ifft, fftfreq
import scipy.io
import matplotlib.pyplot as plt
import math
import random
from matplotlib import pyplot as plt

np.set_printoptions(precision=2, suppress=True)
# Load the .mat file



mat = scipy.io.loadmat("ECG_noise.mat")
data = mat["ecg"]
# print(data)
# print(len(data[0]))
N = 2500
x = np.linspace(0, len(data[0]) - 1, num=N)
# print(len(x))
# plt.plot(x,data[0])
# plt.show()

# Voy a aplicar la FFT para poder ver en que frecuencias esta el ruido

# frec = fft(data)
# print(np.abs(frec))
# print("La long es: " + str(len(abs(frec[0]))))
# plt.plot(x, np.abs(frec[0]))
# plt.show()


# Calculamos la respuesta al escalon unitario de un sistema cuya h(t) es
# una exponencial decreciente
fs = 500 # frecuencia de muestreo
N = 2500 # damos 5 segundos
t = np.arange(0, 2500-1) / fs   # armamos el vector tiempo
print(np.arange(0, 2500) / fs)  # lo veo para que todo_ este bien
tau = 1                         # El factor de la exponencial
h = np.exp(-t / tau)            # La h(t) en funcion del tiempo
delta = np.ones(N)              # el escalon
y = np.convolve(delta, h)       # La respuesta al escalon

#  Graficos:

fig, axis = plt.subplots(2)     # Armo 2 graficos que se van a mostrar juntos (subplots)
fig.suptitle("Graficos")
axis[0].plot(t, h)
axis[0].set_title("h(t)")
axis[1].plot(t, y[1:N])
axis[1].set_title("Respuesta al escalon")
plt.show()
plt.xlabel("Tiempo")
print(delta)

