import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample_poly, hilbert
import adi

# Set SDR parameters
sdr_carrier_freq = 96.9e6  # Choose a local FM station frequency
sdr_rx_gain = 50.0  # Rx gain (Adjust based on signal strength)
M = 15  # Upsampling factor
fs_audio = 44100  # Audio sampling rate (Hz)
fs = M * fs_audio  # PlutoSDR sampling rate (Hz)
N = 900000  # Close to maximum buffer size (1.4 sec per buffer)

# Create PlutoSDR object
sdr = adi.Pluto("ip:192.168.2.1")

# Configure common settings
sdr.sample_rate = int(fs)
sdr.gain_control_mode_chan0 = 'manual'

# Configure Rx
sdr.rx_lo = int(sdr_carrier_freq)
sdr.rx_rf_bandwidth = int(fs)
sdr.rx_buffer_size = int(N)  # Buffer size for Rx samples
sdr.rx_hardwaregain_chan0 = sdr_rx_gain  # Adjust Rx gain (0 to 74.5 dB)

# Clear buffer before capturing data
for _ in range(3):
    sdr.rx()

# Capture multiple frames (about 5 sec of audio)
num_frames = 4
raw_data = []
for _ in range(num_frames):
    temp = sdr.rx()
    raw_data = np.concatenate([raw_data, temp])
raw_data = np.array(raw_data) * 2**-14  # Scale samples to (-1,1)

# Apply 53 kHz LPF to remove out-of-band noise
Nfilt_rf = 11
cutoff_rf = 53e3
b_LPF_rf, a_LPF_rf = butter(Nfilt_rf, cutoff_rf / (fs / 2), btype='low')
filtered_data = filtfilt(b_LPF_rf, a_LPF_rf, raw_data)

# Extract phase from the complex signal
phase_rx = np.unwrap(np.angle(filtered_data))

# Compute the FM demodulated message signal
mRx = (phase_rx[1:] - phase_rx[:-1]) * fs / (2 * np.pi * 75e3)
mRx = np.append(mRx, mRx[-1])  # Append a zero to match the original length

# Extract Audio Components
# Low-pass filter for L+R audio (0-15 kHz)
Nfilt_audio = 9
cutoff_audio = 15e3
b_LPF_audio, a_LPF_audio = butter(Nfilt_audio, cutoff_audio / (fs / 2), btype='low')
L_plus_R = filtfilt(b_LPF_audio, a_LPF_audio, mRx)

# Narrowband filter for pilot tone (19 kHz)
Nfilt_band = 5
cutoff_band_pilot = [18e3, 20e3]
b_narrow, a_narrow = butter(Nfilt_band, np.array(cutoff_band_pilot) / (fs / 2), btype='band')
pilot_rx = filtfilt(b_narrow, a_narrow, mRx)

# Band-pass filter for DSB-SC carrier (38 kHz)
cutoff_band_dsb = [23e3, 53e3]
b_BPF, a_BPF = butter(Nfilt_band, np.array(cutoff_band_dsb) / (fs / 2), btype='band')
DSB_carrier_rx = filtfilt(b_BPF, a_BPF, mRx)

# Extract phase of the recovered pilot using Hilbert transform
pilot_phase = np.unwrap(np.angle(hilbert(pilot_rx)))

# Double the phase to obtain the DSB-SC carrier
DSB_carrier_phase = 2 * pilot_phase

# Recover the difference signal (L-R)
L_minus_R = DSB_carrier_rx * np.cos(DSB_carrier_phase) * 2

# Recover Left and Right channels
L_audio = (L_plus_R + L_minus_R) / 2
R_audio = (L_plus_R - L_minus_R) / 2

# Downsample audio back to 44.1 kHz
L_audio_downsampled = resample_poly(L_audio, 1, M)
R_audio_downsampled = resample_poly(R_audio, 1, M)

# Save the recovered mono and stereo audio as WAV files
from scipy.io.wavfile import write

# Mono Audio (L+R)
mono_audio = (L_audio_downsampled + R_audio_downsampled) / 2
mono_audio_int16 = (mono_audio * 32767).astype(np.int16)  # Convert to 16-bit PCM
write("fm_mono.wav", int(fs_audio), mono_audio_int16)

# Stereo Audio (L, R)
stereo_demod = np.column_stack((L_audio_downsampled, R_audio_downsampled))
stereo_demod_int16 = (stereo_demod * 32767).astype(np.int16)  # Convert to 16-bit PCM
write("fm_stereo.wav", int(fs_audio), stereo_demod_int16)

