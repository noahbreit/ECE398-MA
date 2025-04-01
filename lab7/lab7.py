import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

def rrcosfilter(N, alpha, Tb, Fs):
    """
        Generates a root raised cosine (RRC) filter (FIR) impulse response.

        Parameters
        ----------
        N : int
            Length of the filter in samples.

        alpha : float
            Roll off factor (Valid values are [0, 1]).

        Tb : float
            Symbol period.

        Fs : float
            Sampling Rate.

        Returns
        ---------
        h_rrc : 1-D ndarray of floats
            Impulse response of the root raised cosine filter.
    """

    T_delta = 1/float(Fs)
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
        elif alpha != 0 and t == Tb/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif alpha != 0 and t == -Tb/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Tb) +
                    4*alpha*(t/Tb)*np.cos(np.pi*t*(1+alpha)/Tb))/ (np.pi*t*(1-(4*alpha*t/Tb)*(4*alpha*t/Tb))/Tb)

    return h_rrc


# Create random binary data and BPSK symbols from the previous labs
num_data_symbols = 32
sps = 16

np.random.seed(0)
bits = np.random.randint(0, 2, num_data_symbols) # 0 to 1
bpsk_symbols = bits*2 - 1

############ YOUR CODE STARTS HERE ############
preamble = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1])
# Concatenate preamble, guard interval, and data symbols to create a frame here
guard_interval = np.zeros(len(preamble))
frame = np.concatenate([preamble, guard_interval, bpsk_symbols])
# Upsample and perform pulse-shaping here (the pulse is given)

# Tx_ADC is the pulse-shaped signal

num_taps = 6*sps + 1
pulse = rrcosfilter(N=num_taps, alpha=1.0, Tb=sps, Fs=1)
frame_upsampled = np.zeros(len(frame) * sps)
frame_upsampled[::sps] = frame
Tx_ADC = np.convolve(frame_upsampled, pulse)

############ YOUR CODE ENDS HERE ############


############################################
## Simulate IQ modulator (Tx)
############################################

M = 16  # upsample fs_adc for pass-band simulation
xup = resample_poly(Tx_ADC, M, 1)

  
fs_adc = sps    # sampling rate of ADC
fs_rf = M * fs_adc  # sampling rate for simulating carrier
fc = (M*3/7) * fs_adc # carrier frequency

t = 1/fs_rf*np.arange(len(xup)) # time vector at fs_rf


# u(t): transmitted signal to the channel (passband)
u = np.real(xup) * np.cos(2*np.pi*fc*t) - np.imag(xup) * np.sin(2*np.pi*fc*t)

############################################
## Simulate Channel
############################################
ch_att = 0.1    # channel attenuation

h = np.zeros(M*sps)
h[0] = ch_att
h = np.roll(h, np.random.randint(M*sps))    # random delay

v = np.convolve(u, h) 
noise_amplitude = 0.01
noise = noise_amplitude * np.random.randn(len(v))   # AWGN

v = v + noise

  

############################################
## Simulate IQ demodulator (Rx)
############################################
# Low-Pass Filter (LPF) @ fc
Nfilt = 5
cutoff = fc
b, a = butter(Nfilt, Wn=cutoff, btype='low', fs=fs_rf)

t = 1/fs_rf*np.arange(len(v))

yI = filtfilt(b, a, v*np.cos(2*np.pi*fc*t))
yQ = filtfilt(b, a, v*np.sin(2*np.pi*fc*t))

Rx_ADC = resample_poly(yI + 1j*yQ, 1, M)
############ YOUR CODE ENDS HERE ############
# Tx, Channel, and raw Rx samples from the previous part

############ YOUR CODE STARTS HERE ############
# Matched filtering and symbol timing recovery from the previous labs here
matched_filter = pulse[::-1]
rx_matched = np.convolve(Rx_ADC, matched_filter)
# Frame synchronization: compute cross-correlation and detect the peak  
preamble_upsampled = np.zeros(len(preamble)*sps)
preamble_upsampled[::sps] = preamble
correlation = np.abs(np.correlate(rx_matched, preamble_upsampled, mode='valid')) / len(preamble_upsampled)
# Plot the cross-correlation and its peak
peaks, _ = find_peaks(correlation, height=0.5 * np.max(correlation))
plt.figure(figsize=(10, 5))
plt.plot(correlation, label='Cross-Correlation')
plt.plot(peaks, correlation[peaks], 'rx', label='Peaks')
plt.title('Cross-Correlation and Detected Peaks')
plt.xlabel('Sample Index')
plt.ylabel('Correlation')
plt.legend()
plt.grid(True)
plt.show()
# Plot the IQ constellation
start_idx = peaks[0]  # Assuming first peak corresponds to frame start
rx_data = rx_matched[start_idx:start_idx + len(preamble_upsampled)]  # Example extraction
plt.figure
plt.plot(rx_data.real, rx_data.imag, '.')
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Received Data Constellation')
plt.grid(True)
ax = plt.gca()
ax.set_aspect('equal', adjustable='datalim')
plt.show()
############ YOUR CODE ENDS HERE ############