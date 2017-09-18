import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

sample_rate, signal = scipy.io.wavfile.read('file.wav')  # File assumed to be in the same directory
signal = signal[int(1*sample_rate):int(2 * sample_rate)]  # Keep the first 3.5 sec

#pre-emphasis filter on the signal to amplify the high frequencies (y(t)=x(t)-ax(t-1), keep a between 0.9 t0 1.0)
#Three benefits of pre-emphasis:
# (1) balance the frequency spectrum since high frequencies usually have smaller magnitudes compared to lower frequencies
# (2) avoid numerical problems during the Fourier transform operation and 
#(3) may also improve the Signal-to-Noise Ratio (SNR).

pre_emph = 0.98
esignal = numpy.append(signal[0], signal[1:] - pre_emph * signal[:-1])

#FRAMING
frame_size = 0.030
frame_stride = 0.01
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # gives the frame length and frame step
print frame_length,frame_step
signal_length = len(esignal)
print signal_length
frame_length = int(round(frame_length))
print frame_length
frame_step = int(round(frame_step))
print frame_step
num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
print num_frames
pad_signal_length = num_frames * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_length))
pad_signal = numpy.append(esignal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(numpy.int32, copy=False)]

# Windowing

frames *= numpy.hamming(frame_length)

#fft and power
NFFT = 1024
mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

# Filter-bank

nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = numpy.dot(pow_frames, fbank.T)
filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * numpy.log10(filter_banks)  # dB
fig, ax = plt.subplots()
mfcc= np.swapaxes(filter_banks, 0 ,1)
cax = ax.imshow(filter_banks, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax.set_title('Filterbank')
plt.show()
# MFCC
num_ceps = 12
cep_lifter=12
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
(nframes, ncoeff) = mfcc.shape
n = numpy.arange(ncoeff)
lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
mfcc *= lift  #*
fig1, ax1 = plt.subplots()
mfcc= np.swapaxes(mfcc, 0 ,1)
cax = ax1.imshow(mfcc, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax1.set_title('MFCC')
plt.show()
# Cepstral Mean Normalization

mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)





