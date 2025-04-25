import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, butter, filtfilt

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

# Modulation parameters
sps = 16
Tb = 1
Mary = 16  # BPSK=2, 4-QAM=4, 16-QAM=16
bits_per_symbol = int(np.log2(Mary))
num_data_symbols = 1024*4
num_data_bits = bits_per_symbol*num_data_symbols

np.random.seed(0)
bits = np.random.randint(0, 2, num_data_bits)  # Generate correct number of bits

############ YOUR CODE STARTS HERE ############
# Create QAM symbols
gray_code = [0,1,3,2,4,5,7,6,12,13,15,14,8,9,11,10]
constellation = []
for i in gray_code:
    row = (i // 4) - 1.5
    col = (i % 4) - 1.5
    constellation.append(col + 1j*row)
    
# Normalize constellation (16-QAM normalization factor = 1/sqrt(10))
constellation = np.array(constellation)/np.sqrt(10)
symbol_indices = np.packbits(bits.reshape(-1, bits_per_symbol), 
                           axis=1, bitorder='little').flatten()
mqam_symbols = constellation[symbol_indices]

# Create preamble (BPSK)
barker = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1])
# Concatenate TWO sets of Barkers, guard interval, and data to create a frame here
guard = np.zeros_like(barker) 
frame = np.concatenate((barker, barker, guard, mqam_symbols))
# Upsample and perform pulse-shaping here (the pulse is given)
up_sym = np.zeros(len(frame)*sps)
up_sym[::sps] = frame  # Symbol-spaced upsampling
# Tx_ADC is the pulse-shaped signal

num_taps = 6*sps + 1
pulse = rrcosfilter(N=num_taps, alpha=1.0, Tb=sps, Fs=1)
Tx_ADC = np.convolve(up_sym, pulse)
Tx_ADC = np.concatenate((np.zeros(sps*10), Tx_ADC)) # pad more zeros to help frame sync simulation

############ YOUR CODE ENDS HERE ############

# Rest of Lab9 code for channel simulation and demodulation

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

# Add CFO
N = len(barker)
cfo_limit = 1/(2*N*Tb)
cfo_hz = cfo_limit*0.05

yI = filtfilt(b, a, v*np.cos(2*np.pi*(fc + cfo_hz)*t))
yQ = filtfilt(b, a, -v*np.sin(2*np.pi*(fc + cfo_hz)*t))

Rx_ADC = resample_poly(yI + 1j*yQ, 1, M)
############ YOUR CODE STARTS HERE ############
# Matched filtering and symbol timing recovery from the previous labs here
rx_matched = np.convolve(Rx_ADC, pulse)

# Compute energy for each alignment offset
energy = np.zeros(sps)
for k in range(sps):
    # Compute energy for each offset by summing squared values of sampled segments
    energy[k] = np.sum(np.abs(rx_matched[k::sps])**2)

# Find the best offset
max_ind = np.argmax(energy)

# Align samples using max_ind
rx_aligned = rx_matched[max_ind::sps]

# Self-reference Frame synchronization: compute N-lagged auto-correlation (corr) and detect the peak (frame_ind)
N = len(barker)
corr = np.zeros(len(rx_aligned) - 2*N)
for n in range(len(corr)):
    sum = 0
    for k in range(N):
        sum += np.conj(rx_aligned[n+k]) * rx_aligned[n+k+N]
    corr[n] = np.abs(sum) / N

frame_ind = np.argmax(corr)

# Separate preamble and data symbols
rx_preamble = rx_aligned[frame_ind:frame_ind+2*N]
rx_data = rx_aligned[frame_ind+3*N:frame_ind+3*N+num_data_symbols]

#################################
# Estimate CFO_
#################################

# Calculate delta_phi
products = np.conj(rx_preamble[:N]) * rx_preamble[N:2*N]
sum_product = np.sum(products) / len(products)
delta_phi = np.angle(sum_product)

cfo_hat = delta_phi / (2*np.pi*N*Tb)

#################################
# Correct CFO_
#################################
t_cfo = np.arange(len(rx_aligned)) * Tb
rx_cfo_corrected = rx_aligned * np.exp(-1j*2*np.pi*cfo_hat*t_cfo)
rx_preamble_CFOcor = rx_cfo_corrected[frame_ind:frame_ind+2*N]
rx_data_CFOcor = rx_cfo_corrected[frame_ind+3*N:frame_ind+3*N+num_data_symbols]

#################################
# Recover the bits
#################################
def demodulate(symbols, constellation):
    symbols = symbols * np.sqrt(10)  # Remove normalization
    distances = np.abs(symbols[:, np.newaxis] - constellation)
    symbol_indices = np.argmin(distances, axis=1)
    return np.unpackbits(symbol_indices.astype(np.uint8).reshape(-1,1), 
                       axis=1, bitorder='little')[:, -4:].flatten()

# Use after channel correction
demod_bits = demodulate(rx_data_CFOcor, constellation)

# bit error calculation
bit_err = np.sum(np.abs(bits - demod_bits))
print(f"Bit error: {bit_err}")
print(f"BER (%): {100*bit_err/len(bits):.2f}")
