# Ejercicios de la guia y del libro

import random as rd
from random import seed
from random import randint
import numpy as np
import scipy as sp
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy import signal

# Generador de un array de numeros aleatorios
seed(1)
# generador de nuemeros aleatorios a un arreglo
value = []
i = 0
while i <= 1000:
    val = randint(0, 100)
    value.append(val)
    i = i + 1

max = max(value)

value_nuevo = [x / max for x in value]  # divido todos los valores del array por un numero

from scipy.misc import electrocardiogram

ecg = electrocardiogram()
#print(ecg) # nos muestra una lista con los valores
            #del ECG

fs = 360 #Frec de muestreo
time = np.arange(ecg.size) / fs
plt.plot(time, ecg)
plt.xlabel("time in s")
plt.ylabel("ECG in mV")
plt.xlim(0, 10)
plt.ylim(-5, 4)
plt.show()

# Filtro Butterworth (pasa bajos)

b, a = signal.butter(4, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()


x = np.arange(10, 20)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
# print(x)
# print(y)
r = np.corrcoef(x, y)  # calcula la matrix de correlacion
# print(r)
