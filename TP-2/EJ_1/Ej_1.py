import numpy as np
import scipy.io
import scipy.signal
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from scipy import signal

# 1) Implementar un filtro adaptativo

data = scipy.io.loadmat('/Users/juanmedina1810/PycharmProjects/PSB/TP-2/EJ_1/Fetal.mat')

print(sorted(data.keys()))

# Podemos Elegir entre ALE ==> Adaptive interference supression (Eliminar el ruido de linea, yo se en que frecuencias esta el ruido)
# y ANC ==> Adaptive noise cancellation (Es para frecuencias parasitas, voy aprendiendo cual es el ruido para ir eliminando).

# Vamos a usar el ANC obviamente

print("Cantidad de datos del Feto:", len(data['feto'][0]))
feto = np.array(data['feto'][0])

print("Cantidad de datos del Materno:", len(data['materno'][0]))
materno = data['materno'][0]

print("Fs:", (data['fs'][0][0]))


fs = data['fs'][0][0]
Ts = 1/fs
muestras_totales = len(materno)
t_end = muestras_totales*(1/fs)
N = t_end * fs
L = 500
time = np.arange(0, N) * Ts


def lms(entrada, deseada, L,delta):  # L ==> orden, ir modificando en valor de Delta para obtener el mejor valor posible

    # ERROR CUADRATICO MEDICO

    # La salida es:
    # y = b * x
    # e = deseado - y
    # b = b[-1] + delta * e * x

    # LMS significa error cuadratico medio
    # least mean squares
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

def wiener_hopf(x, d, maxlags):  # x:entrada  d:deseada  maxlags:cantidad de retrasos
    rxx = plt.acorr(x, maxlags=maxlags)[1]  # nos vamos a quedar con el segundo parametro
    rxx = rxx[maxlags:-1]  # nos vamos a quedar con el final
    R = toeplitz(rxx)

    rxy = plt.xcorr(x, d, maxlags=maxlags)[1]  # vector con las correlaciones cruzadas
    rxy = rxy[maxlags:-1]

    R_inv = scipy.linalg.inv(R)
    b = R_inv.dot(rxy)

    return b


# 0 < delta < (1/10*L*px)
a = 0.05  # que tanto pesa lo que aprendi para hacer cada iteracion ! ==> Vamos modificando
px = np.mean(materno ** 2)
delta = a * (1 / (10 * L * px))
print("Px",px)
print("Delta",delta)

b = lms(feto, materno, L, delta)

filtrada = scipy.signal.lfilter(b[0], 1, materno)  # filfilt saca el desfazaje

out = materno - filtrada

plt.plot(time, filtrada)
plt.plot(time, feto)
plt.plot(time, out)
plt.title("ECG")
plt.xlabel("Time [s]")
plt.show()


#plt.figure(figsize=(5,20))
i = 0
titulo = 'ECG(s)'
nombres = ["Materno","Fetal","Filtrada"]
x = [materno,feto,out]

for c in range(len(x)):
  plt.plot(time, x[c]-i,label=nombres[c])
  i = i + 100
plt.title(titulo)
plt.ylabel('Amplitud')
plt.xlabel('Tiempo [s]')
plt.legend()
plt.show()


plt.magnitude_spectrum(materno,Fs=fs,Color='C1',label='Materno')
plt.magnitude_spectrum(feto,Fs=fs,Color='C2',label='Fetal')
plt.magnitude_spectrum(out,Fs=fs,Color='C3',label='Filtrada')
plt.legend()
plt.show()

plt.magnitude_spectrum(materno,Fs=fs,Color='C1',label='Materno',scale='dB')
plt.magnitude_spectrum(feto,Fs=fs,Color='C2',label='Fetal',scale='dB')
plt.magnitude_spectrum(out,Fs=fs,Color='C3',label='Filtrada',scale='dB')
plt.legend()
plt.show()
