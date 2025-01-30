import numpy as np
import matplotlib.pyplot as plt

# Define parameters
frequencies = [5, 10, 15, 20, 25]  # Frequencies of the harmonics in Hz
amplitudes = [16, 8, 4, 2, 1]      # Amplitudes of the harmonics
sampling_rate = 100                # Sampling rate in samples per second

# Time vector: 1 second of signal
t = np.linspace(0, 1, sampling_rate, endpoint=False)

# Initialize the signal
signal = np.zeros(t.size)

#######################
#### Your code here ###

### x(t) == summation( A[:] * sin(2*pi*freq[:]*t) ) ###
# Compute the signal by summing up the harmonics
for i in range(len(frequencies)):
    signal += amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t)

### X(f) == FFT Transform of x(t)
# Frequency domain
spectrum = np.fft.fft(signal)
frequencies = np.fft.fftfreq(t.size, d=1/sampling_rate)
spectrum = np.fft.fftshift(spectrum)
frequencies = np.fft.fftshift(frequencies)

#######################

# Plot the results
plt.figure(figsize=(10, 6))

# Time domain
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()


# # Frequency domain
plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(spectrum))
plt.title('Frequency Domain')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()

plt.tight_layout()
plt.show()



### PART 2 ###
import adi

# Print out ADI version and name
print(adi.__version__)
print(adi.name)

# Default IP address
sdr = adi.Pluto("ip:192.168.2.1")

# Set sampling rate and read back
sdr.sample_rate = int(1e6)
print(sdr.sample_rate)