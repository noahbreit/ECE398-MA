# Import library
import numpy as np
import adi
import matplotlib.pyplot as plt
import os
import time

from math import log2
from scipy.signal import find_peaks

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
    time_idx = ((np.arange(N)-N/2))*T_delta
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

def ascii_to_binary_array(text):
    binary_array = []
    for char in text:
        # Convert each character to its ASCII value, then to binary string
        binary_string = format(ord(char), '08b')
        binary_string = binary_string[::-1]
        # Convert binary string to list of integers and extend the binary_array
        binary_array.extend([int(bit) for bit in binary_string])
    return np.array(binary_array)

def binary_array_to_ascii(binary_array):
    chars = []
    for i in range(0, len(binary_array), 8):
        byte = binary_array[i:i+8]
        byte = byte[::-1]
        char = chr(int(''.join(map(str, byte)), 2))
        chars.append(char)
    return ''.join(chars)

def decimal_to_binary_array(n):
    binary_array = []
    # byte = np.array([int(digit) for digit in bin(n)[2:]])
    binary_string = format(n,'08b')
    binary_string = binary_string[::-1]
    binary_array.extend([int(digit) for digit in binary_string])
    return np.array(binary_array)

def binary_array_to_decimal(binary_array):
    binary_array = binary_array[::-1]
    return int(''.join(map(str, binary_array.tolist())), 2)

# Create radio object
sdr = adi.Pluto("ip:192.168.2.1")

# Debug: True= Enable both Tx and Rx (debugging mode), False= Enable only Rx (demo mode)
Debug = True 

#####################
# Shared parameters #
#####################
MAX_DATA_SYM = 255 # Max data symbol length 
sps = 8 # samples per symbol
fs = 40e6   # sampling rate
symbol_rate = fs/sps
Tb = 1/symbol_rate # symbol period
Mary = 16
bits_per_symbol = int(log2(Mary))

## Preamble
barker = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1])
N = barker.size
preamble = np.concatenate((barker, barker))

HEADER_LEN = int(np.log2((MAX_DATA_SYM+1))//bits_per_symbol)

# Pulse shaping and Matched filter
num_taps = 5*sps+1 #number of taps for rrc
pulse = rrcosfilter(N=num_taps, alpha=1.0, Tb=sps, Fs=1)

# Carrier frequency (find a quiet freq band)
center_freq = 830e6


#####################
# SDR Configuration #
#####################
sdr.sample_rate = int(fs)
if Debug == True:
    # Config Tx
    sdr.tx_lo = int(center_freq)
    sdr.tx_rf_bandwidth = int(fs) # filter cutoff, just set it to the same as sample rate
    sdr.tx_hardwaregain_chan0 = -5 # Increase to increase tx power, valid range is -90 to 0 dB

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(fs)
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 50.0    # initial Rx gain


print("sample_rate (MHz):", sdr.sample_rate*1e-6)
print("tx_lo (MHz):", sdr.tx_lo*1e-6)
print("tx_rf_bandwidth (MHz):", sdr.tx_rf_bandwidth*1e-6)
print("rx_lo (MHz):", sdr.rx_lo*1e-6)
print("rx_rf_bandwidth (MHz):", sdr.rx_rf_bandwidth*1e-6)
print("rx_hardwaregain_chan0:", sdr.rx_hardwaregain_chan0)



#######################################
######### 16QAM Transmitter ###########
#######################################
if Debug == True:
    # This code will run Tx for loop-back

    text = "Hello World!"
    # text = "ECE398MA - Sp25"
    
    # Data bits
    bits = ascii_to_binary_array(text)

    num_data_bits = len(bits)
    num_char = len(text)
    num_data_symbols = int(np.ceil(num_data_bits/bits_per_symbol))

    if num_data_symbols > MAX_DATA_SYM:
        print("Error: Payload (data symbols) size is over the max (255)")
        os._exit(1)
    
    
    print("Tx- Mary: ", Mary)
    print("Tx- bits_per_symbol: ", bits_per_symbol)
    print("Tx- num char: ", num_char)
    print("Tx- num data symbols: ", num_data_symbols)
    print("Tx- num data bits: ", num_data_bits)

    # Header bits
    header_bits = decimal_to_binary_array(num_data_symbols)
    
    # Create a frame: Use your Lab10 code to encode bits to 16QAM
    mqam_symbols = ???
    header_symbols = ???
    guard_interval = np.zeros(N)

    frame = np.concatenate((preamble, guard_interval, header_symbols, mqam_symbols))
    frame_len = frame.size

    # upsample
    up_sym = np.array([], dtype=complex)
    for sym in frame:
        temp = np.zeros(sps, dtype=complex)
        temp[0] = sym
        up_sym = np.concatenate((up_sym, temp))

    Tx_ADC = np.convolve(up_sym, pulse)
    Tx_ADC = np.concatenate((np.zeros(sps*10), Tx_ADC))

    ## Transmit the pulse
    tx_samples = Tx_ADC*2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    print("tx sample size: ", len(tx_samples))

    # Start the transmitter
    sdr.tx_cyclic_buffer = True # Enable cyclic buffers
    sdr.tx(tx_samples) # start transmitting
    
    # Create the IQ constellation
    plt.figure
    plt.subplot(1,2,1)
    plt.plot(mqam_symbols.real, mqam_symbols.imag, '.')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title('Tx 16QAM Symbols')
    plt.grid(True)
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()



####################################
######### 16QAM Receiver ###########
####################################
# Fixed rx buffer size. Increase the size if needed.
sdr.rx_buffer_size = 2*2500


###############################
# Adaptive Gain Control (AGC) #
###############################
target_power_db = -25.0
error_db = 1
current_gain_db = sdr.rx_hardwaregain_chan0
max_gain = 70.0 # PlutoSDR's max Rx gain
min_gain = 0.0 # PlutoSDR's min Rx gain
gain_step = 0.0
max_iter = 10

for i in range (max_iter):
    for k in range (0, 3):
        raw_data = sdr.rx() # dumping a few buffer is needed 

    raw_data = 2**-14*sdr.rx()
    power_db = 10*np.log10(np.mean(np.abs(raw_data)**2))
    gain_step = target_power_db - power_db
    print("iter, rx_gain, power_db", i, sdr.rx_hardwaregain_chan0, power_db)
    ############ YOUR CODE STARTS HERE ############
    if np.abs(gain_step) < error_db:
        break
    if power_db < target_power_db:
        current_gain_db = ???
    elif power_db > target_power_db:
        current_gain_db = ???
        current_gain_db = max(current_gain_db + gain_step, min_gain)
    ############ YOUR CODE ENDS HERE ############  

    sdr.rx_hardwaregain_chan0 = current_gain_db
    time.sleep(1.0)

Rx_ADC = raw_data 

if Debug == True:
    sdr.tx_destroy_buffer()


############ YOUR CODE STARTS HERE ############
# Matched filter from the previous labs here

# Symbol timing recovery from the previous labs here
rx_align = ???

rx_symbols = rx_align[::sps]


#################################
# Frame sync
#################################

# Find the starting index of the frame
start_frame_ind = ???


# Extract rx_preamble from rx_symbols
rx_preamble = rx_symbols[start_frame_ind:???]

#################################
# Estimate CFO
#################################

#################################
# Correct CFO
#################################

######################################
# Estimate/Correct Channel & DC offset
######################################

# Decode rx_header
demod_header_bits = ???
num_data_symbols = binary_array_to_decimal(demod_header_bits)
num_char = int(num_data_symbols*bits_per_symbol/8)

print("Rx- number of ASCII in data: ", num_char)
print("Rx- num_data_symbols: ", num_data_symbols)

# Extract rx_data from rx_symbols
# rx_data: raw data symbols
# rx_data_CFOcor: data symbols after CFO correction
# rx_data_CHcor: data symbols after Channel correction
rx_data = ???
rx_data_CFOcor = ???
rx_data_CHcor = ???

# Create the IQ constellation
plt.figure
plt.subplot(1,2,1)
plt.plot(rx_preamble.real, rx_preamble.imag, '.')
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Preamble sym Before CFO corr')
plt.grid(True)
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.subplot(1,2,2)
plt.plot(rx_data.real, rx_data.imag, '.')
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Data sym Before CFO corr')
plt.grid(True)
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

# Create the IQ constellation
plt.figure
plt.subplot(1,2,1)
plt.plot(rx_preamble_CFOcor.real, rx_preamble_CFOcor.imag, '.')
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Preamble sym After CFO corr')
plt.grid(True)
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.subplot(1,2,2)
plt.plot(rx_data_CFOcor.real, rx_data_CFOcor.imag, '.')
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Data sym After CFO corr')
plt.grid(True)
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()


# Create the IQ constellation
plt.figure
plt.subplot(1,2,1)
plt.plot(rx_preamble_CHcor.real, rx_preamble_CHcor.imag, '.')
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Preamble sym After CH corr')
plt.grid(True)
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.subplot(1,2,2)
plt.plot(rx_data_CHcor.real, rx_data_CHcor.imag, '.')
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Data sym After CH corr')
plt.grid(True)
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

# Decode rx_data
demod_bits = ???

print("Decoded message:")
print(binary_array_to_ascii(demod_bits))


