import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample_poly
import sounddevice as sd


###########################-PART-1-###############################

# Load the audio file
path_to_file = './test_audio.wav'
fs_audio, audio = wavfile.read(path_to_file)

# Display audio sampling rate
print('Audio sampling rate (Hz):', fs_audio)

# Uncomment the following two lines to playback the audio signal
# sd.play(audio.astype(np.float32), int(fs_audio))
# sd.wait() # Wait until the audio is finished playingn  

# Time vector for plotting
t_audio = np.linspace(0, len(audio)/fs_audio, num=len(audio))

# FFT of the audio signal
audio_fft = np.fft.fftshift(np.fft.fft(audio))
freq_audio = np.fft.fftshift(np.fft.fftfreq(len(audio), d=1/fs_audio))

# Plotting time domain and frequency domain
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(t_audio, audio)
plt.title('Time Domain Representation')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(212)
plt.semilogy(freq_audio[:len(freq_audio)], np.abs(audio_fft[:len(audio_fft)]))
plt.title('Frequency Domain Representation')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.savefig("lab1_part1.png")
plt.show()


###########################-PART-2-###############################

# Load audio file
# path_to_file = './test_audio.wav'
# fs_audio, audio = wavfile.read(path_to_file)

# Modulation parameters
fc = 4 * fs_audio  # Carrier frequency
fs = 4 * fc  # New sampling rate

# Resample audio
audio_resampled = resample_poly(audio, fs // fs_audio, 1)
t = np.arange(len(audio_resampled)) / fs

# Generate carrier and modulated signals
carrier = np.cos(2 * np.pi * fc * t)
sc_mod = audio_resampled * carrier  # DSB-SC
tc_mod = (audio_resampled + 1) * carrier  # DSB-TC

# FFT calculations
n = len(sc_mod)
frequencies = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs)[:n])
sc_mod_fft = np.abs(np.fft.fftshift(np.fft.fft(sc_mod)[:n]))
tc_mod_fft = np.abs(np.fft.fftshift(np.fft.fft(tc_mod)[:n]))

# Time domain plot
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:400], audio_resampled[:400], label='Baseband Audio')
plt.plot(t[:400], sc_mod[:400], label='DSB-SC', alpha=0.7)
plt.plot(t[:400], tc_mod[:400], label='DSB-TC', alpha=1.0)
plt.title('First 400 Samples Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Frequency domain plot
plt.subplot(2, 1, 2)
plt.yscale('log')
plt.plot(frequencies, sc_mod_fft, label='DSB-SC')
plt.plot(frequencies, tc_mod_fft, '--', label='DSB-TC')
plt.title('Full Spectrum Analysis')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.tight_layout()
plt.savefig("lab1_part2a.png")
plt.show()

# Zoomed spectrum analysis
plt.figure(figsize=(12, 4))
plt.yscale('log')
plt.plot(frequencies, sc_mod_fft, label='DSB-SC')
plt.plot(frequencies, tc_mod_fft, '--', label='DSB-TC')
plt.xlim(fc-25000, fc+25000)  # Zoom around carrier frequency
plt.title('Spectrum Around Carrier Frequency (Zoomed)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lab1_part2b.png")
plt.show()

#############################-PART-3-############################
# Demodulate using synchronized carrier
phase_offset = 0
carrier_rx = np.cos(2 * np.pi * fc * t + phase_offset)

# Apply low-pass filter to remove high-frequency components
b, a = butter(15, Wn = fs_audio, btype = 'low', fs = fs)

# Demod DSB-SC
sc_demod = sc_mod * carrier_rx  # Demodulate by multiplying with carrier
sc_demod_filt = filtfilt(b, a, sc_demod)  # Apply lowpass filter
sc_demod_filt_fft = np.fft.fftshift(np.fft.fft(sc_demod_filt)[:len(frequencies)])

# Demod DSB-TC
# tc_mod = (audio_resampled + 1) * carrier
tc_demod = np.abs(tc_mod)  # Rectify the received signal
tc_demod_filt = filtfilt(b, a, tc_demod)  # Apply lowpass filter
tc_demod_filt_fft = np.fft.fftshift(np.fft.fft(tc_demod_filt)[:len(frequencies)])

ind_start = 400 
ind_end = 2000 

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title('Original Demodulated Signal')
plt.plot(t[ind_start:ind_end], audio_resampled[ind_start:ind_end], label='Audio') 
plt.plot(t[ind_start:ind_end], sc_demod_filt[ind_start:ind_end], label='DSB-SC')
plt.plot(t[ind_start:ind_end], tc_demod_filt[ind_start:ind_end], label='DSB-TC')
plt.legend() 
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2) 
plt.title('Original Demodulated Freq')
plt.yscale('log')
plt.plot(frequencies, np.abs(sc_demod_filt_fft))
plt.plot(frequencies, np.abs(tc_demod_filt_fft))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.savefig("lab1_part3a.png")
plt.show()

# Playback the demodulated audio signal
sc_demod_resampled = resample_poly(sc_demod_filt, 1, fs // fs_audio)
# sd.play(sc_demod_resampled.astype(np.float32), fs_audio)
# sd.wait() # Wait until the audio finishes playing 

tc_demod_resampled = resample_poly(tc_demod_filt, 1, fs // fs_audio)
# sd.play(tc_demod_resampled.astype(np.float32), fs_audio)
# sd.wait() # Wait until the audio finishes playing

# Create or modify wav content
wavfile.write('sc_demod_resampled.wav', int(fs_audio), sc_demod_resampled.astype(np.float32))
wavfile.write('tc_demod_resampled.wav', int(fs_audio), tc_demod_resampled.astype(np.float32))

# Test with phase offset = π/2
phase_offset_pi2 = np.pi/2
carrier_rx_pi2 = np.cos(2 * np.pi * fc * t + phase_offset_pi2)

# Demod with phase offset
sc_demod_pi2 = sc_mod * carrier_rx_pi2
sc_demod_filt_pi2 = filtfilt(b, a, sc_demod_pi2)
sc_demod_filt_pi2_fft = np.abs(np.fft.fftshift(np.fft.fft(sc_demod_filt_pi2)[:len(frequencies)]))

tc_demod_pi2 = np.abs(tc_mod) # rectified by abs|s(t)|
tc_demod_filt_pi2 = filtfilt(b, a, tc_demod_pi2)
tc_demod_filt_pi2_fft = np.abs(np.fft.fftshift(np.fft.fft(tc_demod_filt_pi2)[:len(frequencies)]))

# Plot both cases
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title('Pi/2 PhaseOffset Demodulated Signal')
plt.plot(t[ind_start:ind_end], sc_demod_filt_pi2[ind_start:ind_end], label='DSB-SC (π/2)')
plt.plot(t[ind_start:ind_end], tc_demod_filt_pi2[ind_start:ind_end], label='DSB-TC (π/2)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Pi/2 PhaseOffset Demodulated Freq')
plt.yscale('log')
plt.plot(frequencies, sc_demod_filt_pi2_fft, label='DSB-SC (π/2)')
plt.plot(frequencies, tc_demod_filt_pi2_fft, '--', label='DSB-TC (π/2)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.tight_layout()
plt.savefig("lab1_part3b.png")
plt.show()

# Playback the demodulated audio signal
sc_demod_pi2_resampled = resample_poly(sc_demod_filt_pi2, 1, fs // fs_audio)
# sd.play(sc_demod_resampled.astype(np.float32), fs_audio)
# sd.wait() # Wait until the audio finishes playing 

tc_demod_pi2_resampled = resample_poly(tc_demod_filt_pi2, 1, fs // fs_audio)
# sd.play(tc_demod_resampled.astype(np.float32), fs_audio)
# sd.wait() # Wait until the audio finishes playing

# Create or modify wav content
wavfile.write('sc_demod_pi2_resampled.wav', int(fs_audio), sc_demod_pi2_resampled.astype(np.float32))
wavfile.write('tc_demod_pi2_resampled.wav', int(fs_audio), tc_demod_pi2_resampled.astype(np.float32))

### NOTES ###
''' I have outputted both DSB-SC and DSB-TC demodulated audio samples and I can hear no difference between them
    when the carrier freq and demodulating freq are exactly equivalent and in-phase.

    This is not true when these frequencies are out-of-phase. DSB-SC is silent when the PhaseOffset is pi/2.
    So... Clearly DSB-TC is more robust when compared to DSB-TC, but DSB-TC requires additional power+complexity
    to transmit the carrier frequency'''