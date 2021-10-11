
import scipy.io
import numpy as np
import scipy.io
import scipy.signal
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from scipy import signal


## ==> GSR
print("long basal", len(gsr[marcas[0]:marcas[1]]))
gsr_basal = gsr[marcas[0]:marcas[1]]
t_start1 = marcas[0]/fs
t_end1 = marcas[1]/fs
N1 = len(gsr_basal)
t_basal1 = np.linspace(t_start1,t_end1,N1)

print("long ansiedad", len(gsr[marcas[1]:marcas[2]]))
gsr_ansiedad = gsr[marcas[1]:marcas[2]]
t_start2 = marcas[1]/fs
t_end2 = marcas[2]/fs
N2 = len(gsr_ansiedad)
t_basal2 = np.linspace(t_start2,t_end2,N2)

# Calculo de la derivada ==> BASAL
print("Start",t_start1)
print("end",t_end1)
dx = (t_end1-t_start1)/N1
print(dx)


DfFD = np.zeros(len(gsr_basal))
for kappa in range(len(gsr_basal)-1):
  DfFD[kappa] =(gsr_basal[kappa +1]-gsr_basal[kappa])/dx

DfFD[-1]=DfFD[-2]
print(DfFD)
print(len(t_basal1))
print(len(gsr_basal))
plt.figure(figsize=(15,9))
plt.plot(t_basal1,DfFD)
plt.plot(t_basal1,gsr_basal)
plt.show()


sin = np.sin(2*np.pi*0.01*t_basal1)

dx = (t_end1-t_start1)/len(x)

print(dx)
DfFD = np.zeros(len(sin))
for kappa in range(len(sin)-1):
  DfFD[kappa] =2*np.pi*2.5*(sin[kappa+1]-sin[kappa])/dx

##### -------

from scipy.fftpack import dst, idst, dct, idct

hatu=dct(gsr_basal,type=2)
#multiply by frequency, minus sign
N = len(gsr_basal)
for i in range(N):
    hatu[i]=-(i)*hatu[i]
#shift to left
hatu[0:N-1]=hatu[1:N]
hatu[N-1]=0.

#dst type III, or equivalently IDST type II.
dotu=idst(hatu,type=2)

dotu=dotu/(2*N)


plt.figure(figsize=(15,9))
plt.plot(t_basal1,br_basal,label='the function')
plt.plot(t_basal1,dotu/(2*np.pi),label='its derivative')
plt.legend()
#plt.figure()
#plt.plot(x,dufft)
plt.show()