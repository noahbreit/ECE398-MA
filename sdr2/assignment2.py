import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample_poly

# Load stereo audio file
path_to_file = 'stereo.wav'  # Update this path
fs_audio, audio = wavfile.read(path_to_file)

# Extract Left and Right audio channels
audio_L = audio[:, 0]
audio_R = audio[:, 1]

# Time vector for original audio
t_audio = np.arange(len(audio_L)) / fs_audio

# Anti-Aliasing Low-Pass Filter (LPF) @ fs_audio (0-15 kHz)
Nfilt = 5
cutoff = 15e3
b_AAF, a_AAF = butter(Nfilt, cutoff / (fs_audio / 2), btype='low')

# Apply LPF to L and R audio signals
audio_L_filt = filtfilt(b_AAF, a_AAF, audio_L)
audio_R_filt = filtfilt(b_AAF, a_AAF, audio_R)

########### PART 2 ###########

# Upsample audio to PlutoSDR sampling rate
M = 15
fs = M * fs_audio  # PlutoSDR sampling rate
mL = resample_poly(audio_L_filt, M, 1)
mR = resample_poly(audio_R_filt, M, 1)

# Time vector for resampled audio
N = len(mL)
t = np.arange(N) / fs

# Pilot Signal
fp = 19e3  # Pilot frequency
ap = 0.1  # Pilot amplitude
pilot = ap * np.cos(2 * np.pi * fp * t)

# DSB-SC Modulated (L-R) Signal
DSB_carrier = np.cos(2 * np.pi * 2 * fp * t)
mLmR_dsb = (mL - mR) * DSB_carrier

# Composite Message Signal m(t)
mTx = (mL + mR) + pilot + mLmR_dsb

# Normalize message signal
mTx /= np.max(np.abs(mTx))

# Plot time-domain samples of m(t)
plt.figure(figsize=(10, 4))
plt.plot(t, mTx)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time-Domain Samples of Message Signal m(t)')
plt.savefig('assignment2a.png')
plt.show()

# Plot log-scale spectrum of m(t)
plt.figure(figsize=(10, 4))
plt.semilogy(np.abs(np.fft.fftshift(np.fft.fft(mTx))))
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.title('Log-Scale Spectrum of Message Signal m(t)')
plt.savefig('assignment2b.png')
plt.show()

######### PART 3 #############

### REUSING ASSIGNMENT 2 CODE ABOVE ###

# FM Modulation Parameters
fc = 100e6  # Carrier frequency
kf = 75e3  # Frequency deviation

# Compute the instantaneous phase
phase = 2 * np.pi * kf * np.cumsum(mTx) / fs

# Complex baseband signal
x_t = np.exp(1j * phase)

# Plot the phase phi(t)
plt.figure(figsize=(10, 4))
plt.plot(t, phase)
plt.xlabel('Time (s)')
plt.ylabel('Phase (rad)')
plt.title('Phase of FM Modulated Signal')
plt.savefig('assignment3a.png')
plt.show()

# Plot the log-scale spectrum of x(t)
plt.figure(figsize=(10, 4))
plt.semilogy(np.abs(np.fft.fftshift(np.fft.fft(x_t))))
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.title('Log-Scale Spectrum of Complex Baseband Signal')
plt.savefig('assignment3b.png')
plt.show()
