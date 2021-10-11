import numpy as np
from scipy import signal as sg
import scipy.io
import math as mt
import matplotlib.pyplot as plt
from scipy import stats

mat = scipy.io.loadmat("neural_data.mat")

x = mat['x']
y = mat['y']

#Calculo el coeficiente de correlacion cruzada

cross_corr = sg.correlate(x[0], y[0])
print(cross_corr)
plt.plot(cross_corr)
plt.show()

# Muestro el punto de correlacion cruzada maxima

print("Retrazo: "+str(max(cross_corr)))