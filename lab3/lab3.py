import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample_poly
import sounddevice as sd

def myHilbert(x):
    """
    Compute the Hilbert transform from scratch
    """
    N = len(x) # assume N is even

    # 1. Compute X(f) by FFT
    X = np.fft.fft(x)
    
    # 2. Set H(f) = -1j sgn(f)
    # Hint: first (second) half of FFT is the positive (negative) frequency
    H = np.empty(N, dtype=complex)
    ############ YOUR CODE STARTS HERE ############
    
    # Set the first half to 1j and second half to -1j   
    H[0] = 0         # DC component is not altered
    H[1:N//2] = -1j  # Positive frequencies
    H[N//2:] = 1j    # Negative frequencies

    ############ YOUR CODE STOPS HERE ############
  

    # 3. Compute Y(f) = H(f)*X(f)
    Y = H*X
    # 4. Compute y(t) by IFFT (remove imaginary part from numerical errors)
    y = np.real(np.fft.ifft(Y))
    
    return y


# Test code for myHilbert
ft = 10
fs = 100
N = 100
  
x = np.cos(2*np.pi*ft/fs*np.arange(N))

x_hat = myHilbert(x)
  
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(x)
plt.subplot(2, 1, 2)
plt.plot(x_hat)
plt.savefig('assignment1.png')
plt.show()

################ PART 2 ##################

# Load the audio file
path_to_file = './test_audio.wav'
fs_audio, audio = wavfile.read(path_to_file)

# Display audio sampling rate
print('Audio sampling rate (Hz):', fs_audio)

# Load "audio" for the test audio before this code

fc = 4 * fs_audio # Carrier frequency
fs = 4 * fc # New sampling rate to satisfy Nyquist

# Compute the Hilbert transform of the original audio
audio_Hilbert = myHilbert(audio)

# Resample the original audio and its Hilbert transform
audio_resampled = resample_poly(audio, fs // fs_audio, 1)
audio_Hilbert_resampled = resample_poly(audio_Hilbert, fs // fs_audio, 1)

t = np.arange(len(audio_resampled)) / fs

# Modulated signal (Upper or Lower)
ssb_mod = audio_resampled * np.cos(2 * np.pi * fc * t) - audio_Hilbert_resampled * np.sin(2 * np.pi * fc * t)
# ssb_mod = audio_resampled * np.cos(2 * np.pi * fc * t) + audio_Hilbert_resampled * np.sin(2 * np.pi * fc * t)

# FFT for modulated signal
n = len(ssb_mod)
ssb_mod_fft = np.abs(np.fft.fftshift(np.fft.fft(ssb_mod)[:n]))
frequencies = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs)[:n])

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:400], audio_resampled[:400], label='Audio')
plt.plot(t[:400], audio_Hilbert_resampled[:400], label='Audio Hilbert')
plt.plot(t[:400], ssb_mod[:400], label='SSB')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2) 
plt.semilogy(frequencies, np.abs(ssb_mod_fft), label='SSB')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.tight_layout()
plt.savefig('assignment2a.png')
plt.show()

# Modulated signal (Upper or Lower)
# ssb_mod = audio_resampled * np.cos(2 * np.pi * fc * t) - audio_Hilbert_resampled * np.sin(2 * np.pi * fc * t)
ssb_mod = audio_resampled * np.cos(2 * np.pi * fc * t) + audio_Hilbert_resampled * np.sin(2 * np.pi * fc * t)

# FFT for modulated signal
n = len(ssb_mod)
ssb_mod_fft = np.abs(np.fft.fftshift(np.fft.fft(ssb_mod)[:n]))
frequencies = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs)[:n])

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:400], audio_resampled[:400], label='Audio')
plt.plot(t[:400], audio_Hilbert_resampled[:400], label='Audio Hilbert')
plt.plot(t[:400], ssb_mod[:400], label='SSB')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2) 
plt.semilogy(frequencies, np.abs(ssb_mod_fft), label='SSB')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.tight_layout()
plt.savefig('assignment2b.png')
plt.show()

############# PART 3 ##############

# Demodulate using synchronized carrier
phase_offset = np.pi / 4
carrier_rx = np.cos(2 * np.pi * fc * t + phase_offset)

# Apply low-pass filter to remove high-frequency components
b, a = butter(15, Wn = fs_audio, btype = 'low', fs = fs)

# Demod SSB
ssb_demod = ssb_mod * np.cos(2 * np.pi * fc * t)  # Demodulate by multiplying carrier_rx signal
ssb_demod_filt = filtfilt(b, a, ssb_demod)  # Pass sc_demod through the low-pass filter
ssb_demod_filt_fft = np.fft.fftshift(np.fft.fft(ssb_demod_filt))

ind_start = 500 
ind_end = 1500 

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t[ind_start:ind_end], audio_resampled[ind_start:ind_end], 'b-', linewidth=5, label='Audio') 
plt.plot(t[ind_start:ind_end], audio_Hilbert_resampled[ind_start:ind_end], 'r-', linewidth=5, label='Audio Hilbert') 
plt.plot(t[ind_start:ind_end], ssb_demod_filt[ind_start:ind_end], 'y-.', linewidth=3, label='SSB Demodulation')
plt.legend() 
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2) 
plt.semilogy(frequencies, np.abs(ssb_demod_filt_fft))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.savefig('assignment3c.png')
plt.show()

# Playback the demodulated audio signal
ssb_demod_resampled = resample_poly(ssb_demod, 1, fs // fs_audio)
sd.play(ssb_demod_resampled.astype(np.float32), int(fs_audio))
sd.wait()  # Wait until the audio is finished playing
