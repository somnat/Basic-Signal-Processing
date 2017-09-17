import matplotlib.pyplot as plt
from scipy.io import wavfile 
from scipy.fftpack import fft
from pylab import *

def FFT(filename):
    fs, samples = wavfile.read(filename) # returns sampling frequency and samples
    c = fft(samples) # returns a list of complex numbers
    print c
    d = len(c)/2  # Using FFT you need only half length 
    plt.plot(abs(c[:(d-1)])) # Takes samples from 0 to d-1
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.show() # shows the plot
    savefig(filename+'.png',bbox_inches='tight') # save the file by file name.
    
FFT('file.wav') # function call

