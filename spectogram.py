import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

def spectogram(filename):
	sample_rate, samples = wavfile.read(filename) # returns sampling rate and samples
	frequencies, times, spectogram = signal.spectrogram(samples, sample_rate) #returns frequencies, times, spectogram
        print spectogram
	plt.pcolormesh(times, frequencies, spectogram)
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.show()

spectogram('time.wav')
