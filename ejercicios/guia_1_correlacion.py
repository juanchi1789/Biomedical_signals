# Los ejercicios que se muestran a continuacion son de la GUIA 1 de PSB
# COREELACION
import numpy as np
from scipy import signal as sg
import scipy.io
import math as mt
import matplotlib.pyplot as plt
from scipy import stats

print("Empecemos la guia !!")

# Ej 1)
x = 10 / 20
amplitud = 0.5 * mt.pow(10, x)  # tambien podemos elevar al exp con **
print("EJ1) --> El valor de la amplitud es: " + str(round(amplitud, 2)) + " Volts")

# Ej 2)
signal = 2.5
noise = 28 * mt.pow(10, -3)
SNR = 20 * mt.log10(signal / noise)
print("Ej2) --> SNR = " + str(round(SNR, 3)) + " dB")

# Ej 3)
# np.random.randn() #asi es como llamo a la funcion randn

# conjunto de datos con la funcion randn

arreglo = np.sort(np.random.randn(15000))  # si unsamos np.random.rand(numero) vamos a tener una distribusion uniforme
x = np.arange(
    500)  # esto lo agregue porque pense que lo podia ver en un plot, es inutil ahora pero lo dejo para acordarme
# para futuros ejercicios
# plt.hist(arreglo, bins="fd") # el numero de "bins" son las clases del histograma
# las clases son la cantidad de espacios para barras
# plt.show()


# Ej 4) Angulo entre 2 señales (ambas representadas como vectores)

sig1 = [1.7, 3, 2.2]
sig2 = [2.6, 1.6, 3.2]

# Asi lo hice yo sin tener idea de nada
abs1 = mt.sqrt(mt.pow(sig1[0], 2) + mt.pow(sig1[1], 2) + mt.pow(sig1[2], 2))
abs2 = mt.sqrt(mt.pow(sig2[0], 2) + mt.pow(sig2[1], 2) + mt.pow(sig2[2], 2))

# Asi calculamos la norma en numpy

# abs1 = np.linalg.norm(sig1)
# abs2 = np.linalg.norm(sig2)

cos = (np.dot(sig1, sig2)) / (abs1 * abs2)

print("Ej4) ---> El angulo es: " + str(round(np.rad2deg(round(mt.acos(cos), 3)), 3)) + " grados")  # hay un round de mas

# podemos visualizar los 2 vectores en 3D

# sig1 = np.array([1.7, 3, 2.2])  # asi se definen. !!
# sig2 = np.array([2.6, 1.6, 3.2])
w = np.array([0, 0, 0])  # el origen de coordenadas
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# ax.set_xlim(0)
# ax.set_ylim(0)
# ax.set_zlim(0)
# ax.quiver(w[0], w[1], w[2], sig1[0], sig1[1], sig1[2])
# ax.quiver(w[0], w[1], w[2], sig2[0], sig2[1], sig2[2])
# plt.show()
# esta bueno pero tengo que mejorarlo

# Ej5) genero 2 funciones sinusoidales y veo si son ortogonales

# onda 1 2hz con 500 puntos
# onda 2 4hz con 500 puntos
# tiempo
t = np.linspace(0, 1, 500)  # de 0 a 1 con 500 puntos equiespaciados
sig1 = np.sin(2 * mt.pi * 2 * t)
sig2 = np.sin(2 * mt.pi * 4 * t)
# plt.plot(t, sig1, 'x')
# plt.plot(t, sig2, 'o')
# plt.show()
# hasta aca tenemos las 2 funciones, tenemos que ver si son ortogonales

corr = np.sum(sig1 * sig2)
# plt.plot(t,(1/500)*(sig1*sig2))
# plt.show()
# print("Pearson coef: "+str(np.corrcoef(sig1,sig2)))
# #como el coef de pearson es muy cercano a cero ==> no hay correlacion
# print("El coeficinete de correlacion es:" +str(corr))
# como es muy cercano a cero, las funciones no estan correlacionadas

# Ej6) tengo que aplicarle el metodo DETREND de matlab a un conjunto de datos

# carga de la señal:
mat = scipy.io.loadmat("data_c1.mat")
data = mat["x"]  # asi se llama el arreglo dentro del .mat
t = np.arange(0, 1000)
# vamos a agregar la linea de tendencia de la media
# tengo que dividir mi data en partes mas pequeñas para poder las
# analizar la media en cada una de esas partes

medias = []  # este es el vector de medias
limite = 0
while limite != 1000:
    media = np.mean(data[limite:limite + 10])
    limite = limite + 10
    medias.append(media)

# plt.plot(t, data) # podemos ver que la data no es estacionaria ya que tiene una tendencia
# lineal hacia arriba
new_data = sg.detrend(data)
print("el desbio anterior: " + str(np.std(medias)))
medias_nuevas = []
limite = 0
while limite != 1000:
    media = np.mean(new_data[limite:limite + 10])
    limite = limite + 10
    medias_nuevas.append(media)
print("EJ6) ---> El desvio nueva: " + str(np.std(medias_nuevas)))
# aca puedo decir que la data que resulta del metodo es estacionaria, ya que
# el desvio es muy bajo, cercano a cero. si miramos el anterior  el desvio es muy alto.
# esta informacion la calculo sobre las medias que tome en intervalos de 100
# Lo hice de esta forma ya que si tengo que ver la estabilidad de una señal miro las medias.


# plt.plot(new_data)
# plt.show()

# Ej7) calcular el coeficinete de pearson de las 2 funciones que esten almacenadas eb el archivo.
# cargamos los datos de las señales
mat = scipy.io.loadmat("Ex2_7.mat")
x = mat['x']
y = mat['y']
# plt.plot(x[0])
# plt.plot(y[0])
# plt.show()
print("Ej7) ----> El Pearson coef: " + str(np.corrcoef(x, y)[0][1]))
# ese valor del coeficiente nos dice que existe una tendencia comun


# Ej8) a) vemos si un coseno y un seno a la misma frec son ortogonales

t = np.linspace(0, 2, 1000)
fx, fy = 1, 1
x = np.sin(2 * np.pi * fx * t)
y = np.cos(2 * np.pi * fy * t)
# plt.plot(t, x)
# plt.plot(t, y)
# plt.show()
# si son ortogonales
print("Ej8) ----> El Pearson coef: " + str(np.corrcoef(x, y)[1][0]))
# dado que el coef de pearson esta tan cercano a cero podemos decir que
# son ortogonales

pearson_coef = []
k = np.arange(1, 100)
contador = 0
for i in k:
    var = k[contador]
    fx = 2
    fy = fx * var
    x = np.sin(2 * np.pi * fx * t)
    y = np.cos(2 * np.pi * fy * t)
    pearson_coef.append(np.corrcoef(x, y)[0][1])
    contador = 1 + contador

# dado los resultados, no estan correlacionadas

# Ej9) #we have to study the correlation between the following signals

mat = scipy.io.loadmat("correl1.mat")
x = mat['x']
y = mat['y']
print("Ej9) ----> The pearson correlation: " + str(np.corrcoef(x, y)[0][1]))
print("Ej9-b) ---> la correlacion normal " + str(round((1 / len(x[0])) * (np.sum(x[0] * y[0])), 3)))
# el valor es modesto, tengo que calcular el angulo entre las funciones
# plt.plot(x[0])
# plt.plot(y[0])
# plt.show()
abs1 = np.linalg.norm(x[0])
abs2 = np.linalg.norm(y[0])
# calculamos el producto punto:
cos = (np.dot(x[0], y[0])) / (abs1 * abs2)
print(
    "Ej10) ---> El angulo es: " + str(round(np.rad2deg(round(mt.acos(cos), 3)), 3)) + " grados")  # hay un round de mas

# Ej11) we are gonna study the correlation between 2 signals of neurons
# se estima que pueden estar relacionadas en una misma funcion pero con un delay

mat = scipy.io.loadmat("neural_data.mat")
x = mat['x']
y = mat['y']
# print(len(x[0]))
# print(len(y[0]))
t = np.linspace(0, 0.2, 1000)
plt.plot(t,x[0])
plt.plot(t,y[0])
plt.show()

# from here without phase displacement

# normal correlation
corr = (1 / len(x[0])) * np.sum(x[0] * y[0])
print("The normal correlation: " + str(corr))
# pearson correlation
pearson_coef = scipy.stats.pearsonr(x[0], y[0])
print("The pearson correlation: " + str(corr))

# from here we search for the phase displacement ==> cross correlation

cross_corr = sg.correlate(x[0], y[0])
cross_corr_1 = sg.correlation_lags(x[0],y[0])
print(cross_corr)
plt.plot(cross_corr)
plt.show()
print("length: "+str((cross_corr)))

# Ej12)
from scipy import signal

mat = scipy.io.loadmat("sines.mat")
x = mat['x']
y = mat['y']
fs = 1 * mt.pow(10, 3)
t = np.linspace(0, 0.5, int(fs))
# plt.plot(t, x[0])
# plt.plot(t, y[0])
# plt.show()

abs1 = np.linalg.norm(x[0])
abs2 = np.linalg.norm(y[0])
# calculamos el producto punto:
cos = (round(np.dot(x[0], y[0]), 2)) / (abs1 * abs2)
print("Ej12) ---> El angulo es: " + str(round(np.rad2deg(mt.acos(cos)), 3)) + " grados")


def lag_finder(y1, y2, sr):
    n = len(y1)

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(
        signal.correlate(y1, y1, mode='same')[int(n / 2)] * signal.correlate(y2, y2, mode='same')[int(n / 2)])

    delay_arr = np.linspace(-0.5 * n / sr, 0.5 * n / sr, n)
    delay = delay_arr[np.argmax(corr)]
    print('y2 is ' + str(delay) + ' behind y1')
    print("The maximum correlation is: " + str(max(corr)))

    plt.figure()
    plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show()


sr = 1024
# lag_finder(x[0], y[0], sr)

# cross_corr = scipy.signal.signaltools.correlate(x, y)[0]
# plt.plot(cross_corr)
# plt.show()
# cross_corr = np.correlate(x[0], y[0],mode = 'full')
# plt.plot(cross_corr)
# plt.show()


# Ej14) Ver si son ortogonales

mat = scipy.io.loadmat("two_var.mat")
x = mat['x']
y = mat['y']
# plt.plot(x[0])
# plt.plot(y[0])
# plt.show()

abs1 = np.linalg.norm(x[0])
abs2 = np.linalg.norm(y[0])
# calculamos el producto punto:
cos = (round(np.dot(x[0], y[0]), 2)) / (abs1 * abs2)
print("Ej14) ---> El angulo es: " + str(round(np.rad2deg(mt.acos(cos)), 3)) + " grados")
# El angulo es 90 grados ==> son ortogonales

# Ej15) trazar las funciones de autocorrelacion de las 2 señales

mat = scipy.io.loadmat("bandwidths.mat")

x = mat['x']  # banda estrecha
y = mat['y']  # banda mas amplia


# plt.plot(x)
# plt.plot(y)
# plt.show()

def autocorr(x):
    resultado = []
    k = 0
    while k < 300:
        long_invers = (1 / len(x))
        resultado.append(long_invers * (np.sum(x * x[1 + k])))
        k = k + 1
    return resultado


res_x = autocorr(x)
res_y = autocorr(y)

# print(len(res_x))
# plt.plot(res_x)
# plt.plot(res_y)
# plt.show()


# Ej 16) Comparar el ECG con sinusoides que varian en frec de 0.25 a 25 hz de incrementos de 0.25

mat = scipy.io.loadmat("eeg_data.mat")
eeg = mat['eeg']  # banda estrecha

t = np.linspace(0, 16.02, 801)
#plt.plot(t, eeg[0])
#plt.show()

# vamos a generar un vector de funciones senos
senos = []
frec = 0

while frec != 25:
    senos.append(np.sin(2*np.pi*frec*t))
    frec = frec + 0.25

long = 801
vect_corr = []
cont = 0
while cont < len(eeg[0]):
    vect_corr[cont].append(np.corrcoef(eeg[0],senos[cont]))
    cont = cont + 1
    print(cont)
