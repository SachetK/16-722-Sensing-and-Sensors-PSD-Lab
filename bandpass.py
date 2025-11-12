import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- PARAMETERS ---
fs = 44100           # Sampling rate
psd_file = "analysis/dataset4_recovered_psd.wav"
am_noisy_file = "datasets/dataset3_am_noisy.wav"
original_file = "datasets/dataset1_base_signal.wav"
fc = 5000             # Carrier frequency
bp_bandwidth = 50     # Bandpass around carrier (Hz)
lowpass_cutoff = 70   # Lowpass after demodulation
order = 4

# --- FUNCTIONS ---
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff/(0.5*fs), btype='low')
    return filtfilt(b, a, data)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut/(0.5*fs), highcut/(0.5*fs)], btype='band')
    return filtfilt(b, a, data)

# --- LOAD SIGNALS ---
psd_recovered, fs1 = sf.read(psd_file)
am_noisy, fs2 = sf.read(am_noisy_file)
psd_recovered = psd_recovered.flatten()
original_signal, _ = sf.read(original_file)
am_noisy = am_noisy.flatten()
t = np.arange(len(psd_recovered)) / fs

# --- FREQUENCY-SELECTIVE FILTERING ---
lowcut = fc - bp_bandwidth
highcut = fc + bp_bandwidth
bp_filtered = butter_bandpass_filter(am_noisy, lowcut, highcut, fs, order)

# Demodulate with known carrier
carrier = np.sin(2 * np.pi * fc * t)
demod = bp_filtered * carrier

# Lowpass to extract baseband
recovered_bp = butter_lowpass_filter(demod, lowpass_cutoff, fs, order)
recovered_bp /= np.max(np.abs(recovered_bp))
psd_recovered /= np.max(np.abs(psd_recovered))

# --- SAVE COMPARISON ---
sf.write("analysis/dataset5_recovered_bandpass.wav", recovered_bp, fs)
print("✅ Saved frequency-selective filtered signal")

# --- PLOT ALL SIGNALS ---
plt.figure(figsize=(12, 10))
plt.subplot(3,1,1)
plt.title("Original Baseband Signal (Dataset 1)")
plt.plot(t[:], original_signal[:])
plt.subplot(3,1,2)
plt.title("PSD Recovered Signal (Dataset 4)")
plt.plot(t[:], psd_recovered[:])
plt.subplot(3,1,3)
plt.title("Frequency-Selective Filter Recovered Signal")
plt.plot(t[:], recovered_bp[:])
plt.tight_layout()
plt.savefig("psd_vs_bandpass_vs_original.pdf")
plt.show()