
# Los ejercicios que se muestran a continuacion son de la GUIA 2 de PSB
# Diseño de filtros digitales

import scipy as sp
from scipy import io
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# EXERCISE 1)
# Diseñe un filter FIR con las siguientes especificaciones:
# Banda de paso  0<=𝑓>=0,25𝜋
# Banda de rechazo  𝑓>0,35𝜋
# Risado  0,95<𝐻(𝑓)<1,05 H(f)<0,1


Delta_min = 20 *np.log10(0.05)
print(Delta_min) # es n dB  ==> error en dB

# determinar el Delta de frec entre la frec de
# stop y la de paso

dF = (0.35 -0.25 ) *np.pi # delta de frecuencia entre las frec de paso
M = 8* np.pi / dF + 1
fc = (0.25 + 0.35) * np.pi / 2  # frecuencia de corte

# Diseñamos el filtro

import matplotlib.pyplot as plt

N = 81  # igual a lo anterior
M = N - 1  #
fc = 0.3 * np.pi

# La funcion es un SINC (seno cardinal)

n = np.arange(0, M)
Hid = np.sin(fc * (n - M / 2)) / (np.pi * (n - M / 2))
Hid[int(M / 2)] = fc / np.pi  # agrego el valor en 40
#Hid = np.sinc(fc*(n-M/2)) #Tenemos que escalarlo
plt.stem(n, Hid)  # funcion de transferencia en tiempo
plt.show()

# programo la ventana

hanning = 0.5 - 0.5 * np.cos(2 * np.pi * n / M)
#plt.stem(hanning)
#plt.show()

# Ahora vamos a hacer el filtro ==> multiplicar la ventana por la Hid

him = hanning * Hid  # filtro en tiempo continuo
#plt.stem(him)
#plt.show()

H_f = np.abs(np.fft.fft(him, 200))
w = np.linspace(0,2 * np.pi, 200)
plt.plot(w[0:100], H_f[0:100])
plt.show()

# le vamos a aplicar el filtro anterior a una señal de ECG

mat = sp.io.loadmat("ECG_noise.mat")
print(mat)
ECG = mat['ecg']
x = ECG[0]# datos
print("aca")
print(ECG.shape[1])
print(len(x))
t0=0
n=2500
ts=1/250
tn=ts*n
t=np.arange(0,tn,ts)
plt.plot(x[0:])
plt.show()
plt.plot(np.abs(np.fft.fft(x[0:],2500*2)))
plt.show()
N=81
M=N-1
fc=0.3
n=np.arange(0,M)
Hid=np.sin(fc*(n-M/2))/(np.pi*(n-M/2))
Hid[int(M/2)]=fc/np.pi
S=np.abs(np.fft.fft(x[0:],2500))/100
H=np.abs(np.fft.fft(Hid,2500))
plt.plot(Hid)
plt.show()
plt.plot(S[0:500])
plt.plot(H[0:500])
plt.show()
fil=np.convolve(x[0:],Hid)
plt.plot(fil)
plt.show()






