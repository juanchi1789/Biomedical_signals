# Cargamos los datos que se encuentran en el archivo spyders.mat
import scipy.io
import numpy as np
import scipy.io
import scipy.signal
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from scipy import signal

mat = scipy.io.loadmat('Spyders.mat')
print(mat.keys())

marcas = mat['MARCAS'][0]
ecg = mat['ecg']
br = mat['br']
gsr = mat['gsr']
fs = mat['fs'][0][0]

# Funciones segmentadas:
ecg_basal = ecg[marcas[0]:marcas[1]]
ecg_ansiedad = ecg[marcas[1]:marcas[2]]


# Vamos a estudiar el periodograma de la derivada de la funcion GSR


t_start1 = marcas[0] / fs
t_end1 = marcas[1] / fs
N1 = len(ecg_basal)
t_basal = np.linspace(t_start1, t_end1, N1)



t_start2 = marcas[1] / fs
t_end2 = marcas[2] / fs
N2 = len(ecg_ansiedad)
t_ansiedad = np.linspace(t_start2, t_end2, N2)



mylist = [1, 2, 3, 4, 5, 6, 7]
N = 3
cumsum, moving_aves = [0], []

for i, x in enumerate(mylist, 1):
    cumsum.append(cumsum[i - 1] + x)
    if i >= N:
        moving_ave = (cumsum[i] - cumsum[i - N]) / N
        # can do stuff with moving_ave here
        moving_aves.append(moving_ave)

print(moving_aves)

plt.plot(mylist)
plt.plot(moving_aves)
plt.show()