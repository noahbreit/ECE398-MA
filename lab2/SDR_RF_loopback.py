from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QLineEdit  # tested with PyQt6==6.7.0
import pyqtgraph as pg # tested with pyqtgraph==0.13.7
import numpy as np
import time
import signal # lets control-C actually close the app
import adi
import logging
import sys

# Defaults
fft_size = 512 # determines buffer size
num_rows = 200
center_freq = 100e6 # 100 MHz Carrier Freq
sample_rates = [61.44, 56, 40, 20, 10, 5, 2, 1, 0.5] # MHz
sample_rate_index = 2
sample_rate = sample_rates[sample_rate_index] * 1e6
time_plot_samples = 500
gain = 50 # 0 to 73 dB. int
epsilon = 1e-10

# SDR setup
sdr = adi.Pluto("ip:192.168.2.1")
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = int(fft_size)
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = gain # dB# Configure Tx parameters

### LOOP-BACK ADDITIONS ###
sdr.tx_rf_bandwidth = int(sample_rate)  # Set filter cutoff to match the sample rate
sdr.tx_lo = int(center_freq)  # Set the carrier frequency
sdr.tx_hardwaregain_chan0 = -50  # Tx gain range (-90 to 0 dB)
###

### GENERATE TWO TONE SIGNAL ###
# Define parameters
Ns = 1000          # Number of samples
# f_tone = 10e6      # Tone frequency (10 MHz)
f_tone = 3e6        # Tone frequency (3 MHz)

############ YOUR CODE STARTS HERE ############

t = np.arange(0, Ns/sample_rate, 1/sample_rate)  # Time vector
# Generate two-tone signal at baseband (hint: cosine already has two tones at f_tone and -f_tone)
tx_samples = np.cos(2*np.pi*f_tone*t)   # Fc Carrier Freq @ 100 MHz, 
                                        # so freq components will appear at
                                        # 90 MHz and 110 MHz

t = np.arange(0, Ns/sample_rate, 1/sample_rate)  # Time vector
# Generate one-tone signal at baseband (110 MHz)
tx_samples = np.exp(1j*2*np.pi*f_tone*t)    # Fc Carrier Freq @ 100 MHz, 
                                            # so freq components will appear at
                                            # 110 MHz

############ YOUR CODE STOPS HERE ############
  
# The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
tx_samples = tx_samples * 2**14  

# Enable cyclic buffer and start transmitting
sdr.tx_cyclic_buffer = True  # Enable cyclic buffers
sdr.tx(tx_samples)  # Start transmitting
###

print("rx_lo (MHz):", sdr.rx_lo*1e-6)
print("fs (MHz):", sdr.sample_rate*1e-6)
print("rx_rf_bandwidth (MHz):", sdr.rx_rf_bandwidth*1e-6)
print("rx_buffer_size:", sdr.rx_buffer_size)
print("rx_hardwaregain_chan0:", sdr.rx_hardwaregain_chan0)

class SDRWorker(QObject):
    def __init__(self):
        super().__init__()
        self.gain = gain
        self.sample_rate = sample_rate
        self.freq = center_freq
        self.spectrogram = -50*np.ones((fft_size, num_rows))
        self.PSD_avg = -50*np.ones(fft_size)

    # PyQt Signals
    time_plot_update = pyqtSignal(np.ndarray)
    freq_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    end_of_run = pyqtSignal() # happens many times a second

    # PyQt Slots
    def update_freq(self, val): # TODO: WE COULD JUST MODIFY THE SDR IN THE GUI THREAD
        sdr.rx_lo = int(val*1e3)
        self.freq = val*1e3

    def update_gain(self, val):
        self.gain = val
        sdr.rx_hardwaregain_chan0 = val

    def update_sample_rate(self, val):
        sdr.sample_rate = int(sample_rates[val] * 1e6)
        sdr.rx_rf_bandwidth = int(sample_rates[val] * 1e6)
        self.sample_rate = int(sample_rates[val] * 1e6)

    # Main loop
    def run(self):
        samples = sdr.rx()*2**-14 # Receive samples
        self.time_plot_update.emit(samples[0:time_plot_samples])

        # avoid dividing by zero issue
        PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2/fft_size + epsilon)

        self.PSD_avg = self.PSD_avg * 0.99 + PSD * 0.01
        self.freq_plot_update.emit(self.PSD_avg)

        self.spectrogram[:] = np.roll(self.spectrogram, 1, axis=1) # shifts waterfall 1 row
        self.spectrogram[:,0] = PSD # fill last row with new fft results
        self.waterfall_plot_update.emit(self.spectrogram)
        
        self.end_of_run.emit() # emit the signal to keep the loop going

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):

        super().__init__()

        self.setWindowTitle("PlutoSDR RF Scanner")
        self.setFixedSize(QSize(900, 600)) 

        self.spectrogram_min = 0
        self.spectrogram_max = 0

        layout = QGridLayout() # overall layout

        # Initialize worker and thread
        self.sdr_thread = QThread()
        self.sdr_thread.setObjectName('SDR_Thread') # so we can see it in htop, note you have to hit F2 -> Display options -> Show custom thread names
        worker = SDRWorker()
        worker.moveToThread(self.sdr_thread)

        # Time plot
        time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'})
        time_plot.setMouseEnabled(x=False, y=True)
        time_plot.setYRange(-1.1, 1.1)
        time_plot_curve_i = time_plot.plot([])
        time_plot_curve_q = time_plot.plot([])
        layout.addWidget(time_plot, 1, 0)

        # Time plot auto range buttons
        time_plot_auto_range_layout = QVBoxLayout()
        layout.addLayout(time_plot_auto_range_layout, 1, 1)
        auto_range_button = QPushButton('Auto Range')
        auto_range_button.clicked.connect(lambda : time_plot.autoRange()) # lambda just means its an unnamed function
        time_plot_auto_range_layout.addWidget(auto_range_button)
        auto_range_button2 = QPushButton('-1 to +1\n(ADC limits)')
        auto_range_button2.clicked.connect(lambda : time_plot.setYRange(-1.1, 1.1))
        time_plot_auto_range_layout.addWidget(auto_range_button2)


        # Layout container for Freq plot related stuff
        freq_plot_layout = QHBoxLayout()
        layout.addLayout(freq_plot_layout, 2, 0)

        # Freq plot
        freq_plot = pg.PlotWidget(labels={'left': 'PSD', 'bottom': 'Frequency [MHz]'})
        imageitem_freq = pg.ImageItem(axisOrder='col-major') # this arg is purely for performance

        freq_plot.setMouseEnabled(x=False, y=True)
        freq_plot_curve = freq_plot.plot([])
        freq_plot.setXRange(center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6)
        freq_plot.setYRange(-30, 20)
        freq_plot_layout.addWidget(freq_plot)

        colorbar_spectrum = pg.HistogramLUTWidget()
        colorbar_spectrum.setImageItem(imageitem_freq) # connects the bar to the waterfall imageitem
        colorbar_spectrum.item.gradient.loadPreset('viridis') # set the color map, also sets the imageitem
        freq_plot_layout.addWidget(colorbar_spectrum)

        # Freq auto range button
        auto_range_button = QPushButton('Auto Range')
        auto_range_button.clicked.connect(lambda : freq_plot.autoRange()) # lambda just means its an unnamed function
        layout.addWidget(auto_range_button, 2, 1)

        # Layout container for waterfall related stuff
        waterfall_layout = QHBoxLayout()
        layout.addLayout(waterfall_layout, 3, 0)

        # Waterfall plot
        waterfall = pg.PlotWidget(labels={'left': 'Samples', 'bottom': 'FFT bins'})
        imageitem = pg.ImageItem(axisOrder='col-major') # this arg is purely for performance

        waterfall.addItem(imageitem)
        waterfall.setMouseEnabled(x=False, y=False)
        waterfall_layout.addWidget(waterfall)

        # Colorbar for waterfall
        colorbar = pg.HistogramLUTWidget()
        colorbar.setImageItem(imageitem) # connects the bar to the waterfall imageitem
        colorbar.item.gradient.loadPreset('viridis') # set the color map, also sets the imageitem
        imageitem.setLevels((-30, 20)) # needs to come after colorbar is created for some reason
        waterfall_layout.addWidget(colorbar)

        # Waterfall auto range button
        auto_range_button = QPushButton('Auto Range\n(-2σ to +2σ)')
        def update_colormap():
            imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
            colorbar.setLevels(self.spectrogram_min, self.spectrogram_max)
        auto_range_button.clicked.connect(update_colormap)
        layout.addWidget(auto_range_button, 3, 1)

        # Freq text input
        freq_text = QLineEdit()
        freq_val = center_freq
        
        freq_button = QPushButton('fc (kHz)')
        freq_label = QLabel()
        def update_freq_label():
            if freq_text.text() == '':
                val = center_freq
            else:
                val = int(freq_text.text())
                # print('text input: ', val)
                worker.update_freq(val)
            freq_label.setText("Frequency [MHz]: " + str(val/1e3))
            freq_plot.autoRange()
        freq_button.clicked.connect(update_freq_label)

        layout.addWidget(freq_text, 4, 0)
        layout.addWidget(freq_button, 4, 1)

        # Gain slider with label
        gain_slider = QSlider(Qt.Orientation.Horizontal)
        gain_slider.setRange(0, 73)
        gain_slider.setValue(int(gain))
        gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        gain_slider.setTickInterval(2)
        gain_slider.valueChanged.connect(worker.update_gain)

        gain_label = QLabel()
        def update_gain_label(val):
            gain_label.setText("Gain: " + str(val))
        gain_slider.valueChanged.connect(update_gain_label)

        update_gain_label(gain_slider.value()) # initialize the label
        layout.addWidget(gain_slider, 5, 0)
        layout.addWidget(gain_label, 5, 1)

        # Sample rate dropdown using QComboBox
        sample_rate_combobox = QComboBox()
        sample_rate_combobox.addItems([str(x) + ' MHz' for x in sample_rates])
        sample_rate_combobox.setCurrentIndex(sample_rate_index) # should match the default at the top
        sample_rate_combobox.currentIndexChanged.connect(worker.update_sample_rate)
        sample_rate_label = QLabel()
        def update_sample_rate_label(val):
            sample_rate_label.setText("Sample Rate: " + str(sample_rates[val]) + " MHz")
        sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)
        update_sample_rate_label(sample_rate_combobox.currentIndex()) # initialize the label
        layout.addWidget(sample_rate_combobox, 6, 0)
        layout.addWidget(sample_rate_label, 6, 1)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Signals and slots stuff
        def time_plot_callback(samples):
            time_plot_curve_i.setData(samples.real)
            time_plot_curve_q.setData(samples.imag)

        def freq_plot_callback(PSD_avg):
            f = np.linspace(worker.freq - worker.sample_rate/2.0, worker.freq + worker.sample_rate/2.0, fft_size) * 1e-6
            freq_plot_curve.setData(f, PSD_avg)
            freq_plot.setXRange(worker.freq*1e-6 - worker.sample_rate/2.0*1e-6, worker.freq*1e-6 + worker.sample_rate/2.0*1e-6)

        def waterfall_plot_callback(spectrogram):
            imageitem.setImage(spectrogram, autoLevels=False)
            sigma = np.std(spectrogram)
            mean = np.mean(spectrogram)
            self.spectrogram_min = mean - 2*sigma # save to window state
            self.spectrogram_max = mean + 2*sigma

        def end_of_run_callback():
            QTimer.singleShot(0, worker.run) # Run worker again immediately

        worker.time_plot_update.connect(time_plot_callback) # connect the signal to the callback
        worker.freq_plot_update.connect(freq_plot_callback)
        worker.waterfall_plot_update.connect(waterfall_plot_callback)
        worker.end_of_run.connect(end_of_run_callback)

        self.sdr_thread.started.connect(worker.run) # kicks off the worker when the thread starts
        self.sdr_thread.start()

app = QApplication([])
window = MainWindow()
window.show() # Windows are hidden by default
signal.signal(signal.SIGINT, signal.SIG_DFL) # this lets control-C actually close the app
app.exec() # Start the event loop
