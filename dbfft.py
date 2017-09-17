import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf

plt.close('all')


def dbfft(x, fs, win=None, ref=32768):
    """
    Calculate spectrum in dB scale
    Args:
        x: input signal
        fs: sampling frequency
        win: vector containing window samples (same length as x).
             If not provided, then rectangular window is used by default.
        ref: reference value used for dBFS scale. 32768 for int16 and 1 for float

    Returns:
        freq: frequency vector
        s_db: spectrum in dB scale
    """

    N = len(x)  # Length of input sequence

    if win is None:
        win = np.ones(1, N)
    if len(x) != len(win):
            raise ValueError('Signal and window must be of the same length')
    x = x * win

    # Calculate real FFT and frequency vector
    sp = np.fft.rfft(x)
    freq = np.arange((N / 2) + 1) / (float(N) / fs)

    # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum.
    s_mag = np.abs(sp) * 2 / np.sum(win)

    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag/ref)

    return freq, s_dbfs


def main():
    # Load the file
    fs, signal = wf.read('file.wav')
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    # Take slice
    N = 8192
    win = np.hamming(N)
    freq, s_dbfs = dbfft(signal[0:N], fs, win)
    freq1,s1_dbfs=dbfft(emphasized_signal[0:N], fs, win)
    # Scale from dBFS to dB
    K = 120
    s_db = s_dbfs + K
    s_db1= s1_dbfs + K
    plt.subplot(2, 1, 1)
    plt.plot(freq, s_db,'r')
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    #plt.show()
    plt.subplot(2, 1, 2)
    plt.plot(freq, s_db1,'g')
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.show()

if __name__ == "__main__":
    main()
