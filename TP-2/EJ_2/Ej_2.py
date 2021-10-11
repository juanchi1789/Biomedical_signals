import spectrum
import scipy.io
import spectrum
import scipy.signal
from scipy.linalg import toeplitz
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams


def zplane(b, a, filename=None):
    """Plot the complex z-plane given a transfer function"""

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b / float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a / float(kd)
    else:
        kd = 1

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn / float(kd)

    # Plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0,
             markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0,
             markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5;
    plt.axis('scaled');
    plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1];
    plt.xticks(ticks);
    plt.yticks(ticks)

    if filename is None:
        plt.title("Diagrama de polos y ceros")
        plt.show()
    else:
        plt.savefig(filename)

    return z, p, k



# Alfa esta entre 8 y 13 hz
# Beta esta entre 20 y 50 hz

alfa_inf = 8
alfa_sup = 13

beta_inf = 20
beta_sup = 50

fc_alfa = alfa_sup-((alfa_sup-alfa_inf)/2)
fc_beta = beta_sup-((beta_sup-beta_inf)/2)

fs = 500
r1 = 0.88
r2 = 0.99
theta = (2*np.pi*fc_alfa)/fs

z1 = r1*np.exp(1j*(2*np.pi*fc_alfa)/fs)
z2 = r1*np.exp(-1j*(2*np.pi*fc_alfa)/fs)
p1 = r2*np.exp(1j*(2*np.pi*fc_alfa)/fs)
p2 = r2*np.exp(-1j*(2*np.pi*fc_alfa)/fs)

num = [1, -2*r1*np.cos(theta), r1**2 ]
den = [1, -2*r2*np.cos(theta), r2**2 ]


sys = signal.TransferFunction(num, den, dt=1/fs)
print(sys)

w, mag, phase = signal.dbode(sys)

plt.semilogx(w, mag)
plt.title("magnitude")
plt.xlabel("frec")
plt.show()

plt.semilogx(w, phase)
plt.title("Phase magnitude plot")
plt.show()

zplane(num, den) # POLOS Y CEROS


mean = 0
std = 1
num_samples = 10000
samples = np.random.normal(mean, std, size=num_samples)
plt.plot(samples)
plt.title("Ruido blanco")
plt.show()
y = scipy.signal.filtfilt(num, den, samples)
plt.magnitude_spectrum(y)
plt.show()

# <------------------------------------------> HERE

# CREACION DE RUIDO BLANCO

fs = 500
tfinal = 2
N = fs*tfinal

time = np.linspace(0, tfinal, N)
rb = np.random.randn(N)
plt.figure(figsize=(20, 5))
plt.title('RUIDO BLANCO')
plt.xlabel('Tiempo [seg]')
plt.ylabel('Amplitud')
plt.plot(time, rb)
plt.show()

ritmos_cerebrales = {
    "alpha": [8,13],
    "beta": [20,50]
}

fc1 = 10.5 #FREC CENTRAL ALFA
fc2 = 35 #FREC CENTRAL BETA
fs = 500
tfinal = 2
N = fs*tfinal

# CALCULOS THETA ALFA Y BETA

theta1 = 2*np.pi*fc1 / fs
theta2 = 2*np.pi*fc2 / fs
print('Theta Beta =', theta1)
print('Theta Alfa =', theta2)

# CREAMOS FUNCION PARA OBTENER b Y a

def hz(r1,r2,theta):
  num = np.array([1, -2*r1*np.cos(theta), r1**2 ]) # CEROS r2
  den = np.array([1, -2*r2*np.cos(theta), r2**2 ]) # POLOS r1
  return den,num

# ---SINTESIS RITMO BETA---

denBeta, numBeta = hz(0.998,0.282, theta2)
zplane(denBeta,numBeta)

ruidoBeta = scipy.signal.filtfilt(denBeta, numBeta, rb)
espectroBeta = abs(scipy.fft.fft(ruidoBeta))/N

f = np.linspace(0,1,N)*fs
plt.figure(figsize=(20,5))
plt.plot(f,espectroBeta)
plt.title('BETA')
plt.xlim([0,100])
plt.show()

# ---SINTESIS RITMO ALFA---
denAlfa, numAlfa = hz(0.985, 0.271, theta1)
zplane(denAlfa,numAlfa)

ruidoAlfa = scipy.signal.filtfilt(denAlfa, numAlfa, rb)
espectroAlfa = abs(scipy.fft.fft(ruidoAlfa))/N

f = np.linspace(0,1,N)*fs
plt.figure(figsize=(20,5))
plt.plot(f,espectroAlfa)
plt.title('ALFA')
plt.xlim([0,100])
plt.show()

# ¿Como responde nuestro sistema al ruido blanco?

def hz(r1,r2,theta):
  num = np.array([1, -2*r1*np.cos(theta), r1**2 ]) # CEROS r2
  den = np.array([1, -2*r2*np.cos(theta), r2**2 ]) # POLOS r1
  return den,num

denAlfa, numAlfa = hz(0.985, 0.271, theta1)
zplane(denAlfa,numAlfa)

ruidoAlfa = scipy.signal.filtfilt(denAlfa, numAlfa, rb)
espectroAlfa = abs(scipy.fft.fft(ruidoAlfa))/N

f = np.linspace(0,1,N)*fs
plt.figure(figsize=(20,5))
plt.plot(f,espectroAlfa)
plt.title('ALFA--Mio')
plt.xlim([0,100])
plt.show()

# SUMA DE ESPECTROS
suma = espectroAlfa + espectroBeta
plt.figure(figsize=(20,5))
plt.plot(f,suma)
plt.title('SUMA')
plt.xlim([0,100])
plt.show()

# equivalente FIR

ritmos_cerebrales = {
    "alpha": [8,13],
    "beta": [20,50]
}

# --------forma 1
L = 50
k = (L-1)/2
fh = 20
fl = 50

'''
for i in range(-(L-1)/2 , (L-1)/2):
  bk[i] = (np.sin(2*np.pi*fh*i) - np.sin(2*np.pi*fl*i))/(np.pi*i)
return bk'''

# --------forma2
cutoff = ritmos_cerebrales['beta']

b = scipy.signal.firwin(L+1, cutoff, fs=fs, pass_zero='bandpass')
print('Coeficiente del FIR:',b)

H = np.fft.fft(b)

x_filtrada = scipy.signal.lfilter(b, 1, rb)

plt.figure(figsize = (20,10))
plt.plot(time,rb)
plt.plot(time,x_filtrada - 5)
plt.show()

# PLOTEO EL FILTRO
# --------- 1
print('forma 1 de plotear el filtro')
plt.figure()
plt.plot(b,'bo-', linewidth=2)
w, h = scipy.signal.freqz(b,1)

plt.figure()
plt.plot(w,h)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.title('Frequency Response')
plt.show()

# --------- 2
print('forma 2 de plotear el filtro')
sys = signal.TransferFunction(b, 1)
w, mag, phase = signal.bode(sys)
plt.figure()
plt.semilogx(w, mag)
plt.figure()
plt.semilogx(w, phase)
plt.show()



