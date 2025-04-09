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

## Assignment 1 Implementation
def assignment1():
    delta_f = 1/Tb
    
    # Generate signals
    phi0 = generate_2fsk_signal([0], fc, delta_f)
    phi1 = generate_2fsk_signal([1], fc, delta_f)
    
    # Time domain plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(t[:100], phi0[:100], label='Phi_0 (0)')
    plt.plot(t[:100], phi1[:100], label='Phi_1 (1)')
    plt.title('First 100 Samples')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    
    # Frequency domain plot
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

## Detection Implementations
def coherent_detection_2fsk(rx_signal, fc, delta_f, theta):
    f0 = fc - delta_f/2
    f1 = fc + delta_f/2
    phi0 = np.sqrt(2)*np.cos(2*np.pi*f0*t + theta[0])
    phi1 = np.sqrt(2)*np.cos(2*np.pi*f1*t + theta[1])
    
    ys = []
    syms = []
    for i in range(0, len(rx_signal), N):
        seg = rx_signal[i:i+N]
        if len(seg) < N: continue
        
        y0 = np.dot(seg, phi0)/fs
        y1 = np.dot(seg, phi1)/fs
        ys.append([y0, y1])
        
        dist0 = np.linalg.norm([y0-1, y1])
        dist1 = np.linalg.norm([y0, y1-1])
        syms.append(0 if dist0 < dist1 else 1)
    
    return np.array(ys), np.array(syms)

def noncoherent_detection_2fsk(rx_signal, fc, delta_f, theta):
    f0 = fc - delta_f/2
    f1 = fc + delta_f/2
    
    # Create quadrature basis
    phi0_I = np.sqrt(2)*np.cos(2*np.pi*f0*t + theta[0])
    phi0_Q = np.sqrt(2)*np.sin(2*np.pi*f0*t + theta[0])
    phi1_I = np.sqrt(2)*np.cos(2*np.pi*f1*t + theta[1])
    phi1_Q = np.sqrt(2)*np.sin(2*np.pi*f1*t + theta[1])
    
    ys = []
    syms = []
    for i in range(0, len(rx_signal), N):
        seg = rx_signal[i:i+N]
        if len(seg) < N: continue
        
        # Compute projections
        y0_I = np.dot(seg, phi0_I)/fs
        y0_Q = np.dot(seg, phi0_Q)/fs
        y1_I = np.dot(seg, phi1_I)/fs
        y1_Q = np.dot(seg, phi1_Q)/fs
        
        mag0 = np.sqrt(y0_I**2 + y0_Q**2)
        mag1 = np.sqrt(y1_I**2 + y1_Q**2)
        ys.append([mag0, mag1])
        
        dist0 = np.linalg.norm([mag0-1, mag1])
        dist1 = np.linalg.norm([mag0, mag1-1])
        syms.append(0 if dist0 < dist1 else 1)
    
    return np.array(ys), np.array(syms)

## Assignment 2 Experiments
def run_experiments():
    # Common parameters
    num_syms = 128
    syms = np.random.randint(0, 2, num_syms)
    noise_amp = 0.1
    
    # Phase offset experiment
    thetas = [
        [0, 0],
        [np.pi/2, np.pi/2],
        np.random.uniform(-np.pi, np.pi, 2),
        np.random.uniform(-np.pi, np.pi, 2),
        np.random.uniform(-np.pi, np.pi, 2)
    ]
    
    for theta in thetas:
        signal = generate_2fsk_signal(syms, fc, 1/Tb)
        noise = noise_amp * np.random.randn(len(signal))
        rx_signal = signal + noise
        
        y_coh, sym_coh = coherent_detection_2fsk(rx_signal, fc, 1/Tb, theta)
        y_noncoh, sym_noncoh = noncoherent_detection_2fsk(rx_signal, fc, 1/Tb, theta)
        
        ser_coh = 100*np.mean(syms != sym_coh)
        ser_noncoh = 100*np.mean(syms != sym_noncoh)
        
        print(f"\nTheta: {theta}")
        print(f"Coherent SER: {ser_coh:.2f}%")
        print(f"Non-coherent SER: {ser_noncoh:.2f}%")
        
        # Constellation plot
        plt.figure()
        plt.plot(y_coh[:,0], y_coh[:,1], 'o', alpha=0.5, label='Coherent')
        plt.plot(y_noncoh[:,0], y_noncoh[:,1], 'o', alpha=0.5, label='Non-coherent')
        plt.grid()
        plt.axis('equal')
        plt.title(f'Constellation (Theta={theta})')
        plt.legend()
        plt.show()

    # Frequency separation experiment
    delta_fs = [1/Tb, 0.5/Tb, 10.5/Tb]
    for df in delta_fs:
        signal = generate_2fsk_signal(syms, fc, df)
        noise = noise_amp * np.random.randn(len(signal))
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

## Main Execution
if __name__ == "__main__":
    assignment1()
    run_experiments()
