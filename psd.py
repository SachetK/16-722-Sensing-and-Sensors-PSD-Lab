import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- PARAMETERS ---
fs = 44100
cutoff = 70.0   # lowpass cutoff (Hz), should be above message frequency (60 Hz)
order = 5

# --- LOAD DATA ---
am_noisy_rec, fs1 = sf.read("recorded/dataset3_am_noisy_recorded.wav")
carrier_ref, fs2 = sf.read("datasets/dataset4_reference.wav")
base_ref, fs3 = sf.read("recorded/dataset1_base_signal_recorded.wav")

# Ensure matching lengths
n = min(len(am_noisy_rec), len(carrier_ref))
am_noisy_rec = am_noisy_rec[:n]
carrier_ref = carrier_ref[:n]

# --- PHASE-SENSITIVE DETECTION ---
mixed = am_noisy_rec.flatten() * carrier_ref.flatten()  # multiply (mixing)

# --- LOWPASS FILTER ---
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
    return filtfilt(b, a, data)

recovered = butter_lowpass_filter(mixed, cutoff, fs)

# --- NORMALIZE ---
recovered /= np.max(np.abs(recovered))
base_ref /= np.max(np.abs(base_ref))

# --- SAVE recovered result ---
sf.write("analysis/dataset4_recovered_psd.wav", recovered, fs)
print("✅ Saved dataset4_recovered_psd.wav")

# --- PLOTS ---
t = np.arange(len(recovered)) / fs

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.title("Original Baseband Signal (Dataset 1)")
plt.plot(t[:], base_ref[:])
plt.subplot(3, 1, 2)
plt.title("Recorded AM + Noise (Dataset 3 Recorded)")
plt.plot(t[:], am_noisy_rec[:])
plt.subplot(3, 1, 3)
plt.title("Recovered Signal after PSD (Dataset 4)")
plt.plot(t[:], recovered[:])
plt.tight_layout()

# --- SAVE FIGURE AS PDF ---
plt.savefig("psd_recovery_plot.pdf")  # saves current figure to PDF
print("✅ Saved psd_recovery_plot.pdf")

plt.show()