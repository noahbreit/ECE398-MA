import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

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

# Plot time-domain samples of the Left and Right channels
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(t_audio, audio_L, label='Original Left Channel')
plt.plot(t_audio, audio_L_filt, label='Filtered Left Channel')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time-Domain Samples of Left Channel')
plt.legend()

plt.subplot(122)
plt.plot(t_audio, audio_R, label='Original Right Channel')
plt.plot(t_audio, audio_R_filt, label='Filtered Right Channel')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time-Domain Samples of Right Channel')
plt.legend()
plt.tight_layout()
plt.savefig('assignment1a.png')
plt.show()

# Plot log-scale spectrum of the original and filtered Left-channel audio
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.semilogy(np.abs(np.fft.fftshift(np.fft.fft(audio_L))), label='Original Left Channel')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.title('Log-Scale Spectrum of Original Left Channel')
plt.legend()

plt.subplot(122)
plt.semilogy(np.abs(np.fft.fftshift(np.fft.fft(audio_L_filt))), label='Filtered Left Channel')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.title('Log-Scale Spectrum of Filtered Left Channel')
plt.legend()
plt.tight_layout()
plt.savefig('assignment1b.png')
plt.show()
