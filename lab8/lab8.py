import numpy as np
import matplotlib.pyplot as plt

## Parameters
fs = 1000  # Sampling rate (Hz)
fc = 100   # Carrier frequency (Hz)
Tb = 1     # Symbol duration (s)
N = int(fs * Tb)  # Samples per symbol
t = np.linspace(0, Tb, N, endpoint=False)

## FSK Signal Generation
def generate_2fsk_signal(syms, fc, delta_f):
    signal = []
    for sym in syms:
        f = fc + (2*sym - 1)*delta_f/2  # Calculate frequency
        s = np.sqrt(2)*np.cos(2*np.pi*f*t)
        signal.extend(s)
    return np.array(signal)

######################## PART ONE ##############################
delta_f = 1/Tb
phi0 = generate_2fsk_signal([0], fc, delta_f)
phi1 = generate_2fsk_signal([1], fc, delta_f)

# matplotlib time domain plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t[:100], phi0[:100], label='Phi_0 (0)')
plt.plot(t[:100], phi1[:100], label='Phi_1 (1)')
plt.title('First 100 Samples')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

# matplotlib freq domain plot
plt.subplot(1, 2, 2)
for sig, label in [(phi0, 'Phi_0'), (phi1, 'Phi_1')]:
    fft_result = np.fft.fftshift(np.fft.fft(sig))
    freq = np.fft.fftshift(np.fft.fftfreq(len(sig), 1/fs))
    plt.plot(freq, np.abs(fft_result)/len(sig), label=label)

plt.xlim(50, 150)
plt.title('Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


######################## PART TWO ############################
num_syms = 128
syms = np.random.randint(0, 2, num_syms)

# Generate FSK signal
delta_f = (1.0 / Tb)
signal = generate_2fsk_signal(syms, fc, delta_f)

# Add AWGN
noise_amplutide = 0.1
noise = noise_amplutide * np.random.randn(len(signal))
rx_signal = signal + noise

## Detection Implementations
def coherent_detection_2fsk(rx_signal, fc, delta_f, theta):
    ys = np.empty((0,2))    # observation vectors
    syms = np.array([])     # recoered symbols (0 or 1)

    # create two orthonormal basis
    f0 = fc - delta_f/2
    f1 = fc + delta_f/2
    phi0 = np.sqrt(2)*np.cos(2*np.pi*f0*t + theta[0])
    phi1 = np.sqrt(2)*np.cos(2*np.pi*f1*t + theta[1])

    for i in range(0, len(rx_signal), N):
        # received signal in Tb duration
        segment = rx_signal[i:i+N]
        # if len(segment) < N: continue
        
        # compute the y vector coefficients
        y0 = np.dot(segment, phi0)/fs
        y1 = np.dot(segment, phi1)/fs

        y = np.array([y0, y1])
        ys = np.vstack([ys, y])
        
        # make decisions by choosing the closest distance (use linalg.norm)
        SYMBOL_ZERO = np.array([-1, 0])
        SYMBOL_ONE = np.array([1, 0])
        decision = np.argmin([np.linalg.norm(y - SYMBOL_ZERO), np.linalg.norm(y - SYMBOL_ONE)])
        syms = np.append(syms, decision)
    
    return ys, syms

def noncoherent_detection_2fsk(rx_signal, fc, delta_f, theta):
    # TODO
    
    return ys, syms

# Phase offset experiment
thetas = [
    [0, 0],
    [np.pi/2, np.pi/2],
    np.random.uniform(-np.pi, np.pi, 2),
    np.random.uniform(-np.pi, np.pi, 2),
    np.random.uniform(-np.pi, np.pi, 2)
]

for theta in thetas:
    print(f"\nTheta: {theta}")
    y_coh, sym_coh = coherent_detection_2fsk(rx_signal, fc, delta_f, theta)
    y_noncoh, sym_noncoh = noncoherent_detection_2fsk(rx_signal, fc, delta_f, theta)
    
    ser_coh = 100*np.mean(syms != sym_coh)
    ser_noncoh = 100*np.mean(syms != sym_noncoh)

    print('Symbol error rate (coherent demod) (%): ', 100*np.count_nonzero(syms != sym_coh)/num_syms)
    print('Symbol error rate (non-coherent demod) (%): ', 100*np.count_nonzero(syms != sym_noncoh)/num_syms)
    
    # Constellation plot
    plt.figure()
    plt.plot(y_coh[:,0], y_coh[:,1], 'o', label='coherent')
    plt.plot(y_noncoh[:,0], y_noncoh[:,1], 'o', label='non-coherent')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.grid()
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel(r'$\mathrm{\phi_0(t)}$')
    plt.ylabel(r'$\mathrm{\phi_1(t)}$')
    plt.title('Constellation')
    plt.show()

# Frequency separation impact
delta_fs = [1.0/Tb, 1.0/(2*Tb), 10.5/Tb]
for df in delta_fs:
    signal = generate_2fsk_signal(syms, fc, df)
    noise = noise_amplutide * np.random.randn(len(signal))
    rx_signal = signal + noise
    
    _, sym_noncoh = noncoherent_detection_2fsk(rx_signal, fc, df, [0,0])
    y_noncoh, _ = noncoherent_detection_2fsk(rx_signal, fc, df, [0,0])
    
    plt.figure()
    plt.plot(y_noncoh[:,0], y_noncoh[:,1], 'o')
    plt.title(f'Non-coherent Constellation (delta_f={df:.2f} Hz)')
    plt.xlabel('Phi_0(t)')
    plt.ylabel('Phi_1(t)')
    plt.grid()
    plt.axis('equal')
    plt.show()

