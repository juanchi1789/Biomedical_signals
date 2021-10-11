import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


#H(z)=(0.2+0.5*z^-1)/(1-0.2*z^-1+0.8*z^-2)
from scipy.signal.ltisys import TransferFunctionDiscrete

t_fin = 512/1000

t = np.linspace(0,t_fin,1000)
print(len(t))

sys = signal.TransferFunction([0.2,0.5,0], [1, -0.2, 0.8], dt=t_fin)
w, mag, phase = sys.bode()
plt.figure()
plt.semilogx(w, mag) # Bode magnitude plot
plt.grid()
plt.show()
print(type(mag))
plt.stem(np.diff(mag))
plt.show()



