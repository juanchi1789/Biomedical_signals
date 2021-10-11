from spectrum import arma_estimate, arma2psd, marple_data
import pylab
import matplotlib.pyplot as plt

a,b, rho = arma_estimate(marple_data, 15, 15, 30)
psd = arma2psd(A=a, B=b, rho=rho, sides='centerdc', norm=True)
pylab.plot(10 * pylab.log10(psd))
pylab.ylim([-50,0])
plt.show()

import spectrum
import numpy as np
import scipy.io
import scipy.signal
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from scipy import signal
fs = 30.0
time = np.arange(0, 10, 1/fs)

x = np.sin(2*np.pi*1*time)+np.sin(2*np.pi*4*time)
plt.plot(time,x)
plt.show()
plt.magnitude_spectrum(x,Fs=fs,Color='C1')
plt.show()

a,b, rho = spectrum.arma_estimate(x, 2, 2, 4)
print("A:",a)
print("B:",b)
psd = spectrum.arma2psd(A=a, B=b, rho=rho, sides='centerdc', norm=True)
plt.plot(20*np.log10(psd/max(psd)))
#plt.axis([0, 4096, -80, 0])
plt.xlabel('Frequency')
plt.ylabel('power (dB)')
plt.grid(True)
plt.show()

fourier_transform = np.fft.rfft(x)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, fs/2, len(power_spectrum))
plt.plot(frequency, power_spectrum)
plt.show()

from spectrum import arma_estimate, arma2psd, marple_data
import pylab

#a,b, rho = arma_estimate(marple_data, 15, 15, 30)
psd = arma2psd(A=a, B=b, rho=rho, sides='centerdc', norm=True)
pylab.plot(psd)
plt.show()




fourier_transform = np.fft.rfft(x)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, fs/2, len(power_spectrum))
plt.plot(frequency, power_spectrum)
plt.show()

a,b, rho = spectrum.arma_estimate(x, 2, 2, 4)
#a, rho =spectrum.ma(x,2,4)
print("A:",a)
print("B:",b)
print("Rho:",rho)

psd = spectrum.arma2psd(A=a, B=b, rho=rho, sides='centerdc', norm=True)
#psd = spectrum.arma2psd(A=a, rho=rho, sides='centerdc', norm=True)
frequency = np.linspace(0, fs/2, len(psd))

#plt.plot(frequency,20*np.log10(psd/max(psd)))
#plt.plot(frequency,psd)
plt.magnitude_spectrum(psd,Fs=fs,Color='C1')

plt.xlabel('Frequency')
plt.ylabel('power (dB)')
plt.grid(True)
plt.show()

print("Polos:",(signal.TransferFunction(b, a, dt=fs).poles))

z,p,k = scipy.signal.tf2zpk(b, a)

plt.scatter(np.real(p),np.imag(p))
plt.scatter(np.real(z),np.imag(z))
plt.show()

mean = 0
std = 1
num_samples = 10000
samples = np.random.normal(mean, std, size=num_samples)
plt.plot(samples)
plt.title("Ruido blanco")
plt.show()
y = scipy.signal.filtfilt(num,den,samples)
print(len(y))
print(len(w))
plt.magnitude_spectrum(y)
plt.show()

# Ritmos Alfa y Beta


alfa = [8,13]
beta = [20,50]

# --- Para el ritmo Alfa

orden = 60 # Grado arbitrario
cutoff = alfa

b = scipy.signal.firwin(orden, cutoff, fs=fs, pass_zero='bandpass')
print('Coeficiente del filtro FIR:',b)

H = np.fft.fft(b)
x_filtrada = scipy.signal.lfilter(b, 1, rb)

plt.figure(figsize = (20,10))
plt.plot(time,rb)
plt.plot(time,x_filtrada - 5)
plt.title("Funcion original y filtrada para Alfa")
plt.show()


# --- Para el ritmo Beta

orden = 60 # Grado arbitrario
cutoff = beta

b = scipy.signal.firwin(orden, cutoff, fs=fs, pass_zero='bandpass')
print('Coeficiente del filtro FIR:',b)

H = np.fft.fft(b)
x_filtrada = scipy.signal.lfilter(b, 1, rb)

plt.figure(figsize = (20,10))
plt.plot(time,rb)
plt.plot(time,x_filtrada - 5)
plt.title("Funcion original y filtrada para Beta")
plt.show()

# ¿Cual es el orden optimo del filtro? ==> Criterio de Akaike
sampling_rate = 500
data = rb
fourier_transform = np.fft.rfft(x_filtrada)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
plt.plot(frequency, power_spectrum)
plt.show()