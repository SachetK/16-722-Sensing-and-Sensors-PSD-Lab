import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# --- SETTINGS ---
fs = 44100        # Sampling rate (Hz)
duration = 10.0    # seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# --- Frequencies ---
fc = 5000   # carrier frequency (Hz)
fsig = 60   # baseband (signal) frequency (Hz)
noise_amp = 0.4

envelope_freq = 1  # 1 Hz amplitude variation
envelope = 0.5 * (1 + np.sin(2 * np.pi * envelope_freq * t))  # goes from 0 to 1

# Baseband signal (message)
signal = envelope * np.sin(2 * np.pi * fsig * t)

carrier = np.sin(2 * np.pi * fc * t)           # high-frequency carrier

# --- DATASET 1: Clean baseband signal (for reference) ---
sf.write("datasets/dataset1_base_signal.wav", signal, fs)
print("✅ Saved dataset1_base_signal.wav")

# --- DATASET 2: Baseband signal + noise ---
signal_noisy = signal + noise_amp * np.random.randn(len(t))
sf.write("datasets/dataset2_signal_noisy.wav", signal_noisy, fs)
print("✅ Saved dataset2_signal_noisy.wav")

# --- DATASET 3: Amplitude-modulated (true PSD input) + noise ---
am_signal = signal * carrier                   # multiplicative modulation
am_noisy = am_signal + noise_amp * np.random.randn(len(t))
sf.write("datasets/dataset3_am_noisy.wav", am_noisy, fs)
print("✅ Saved dataset3_am_noisy.wav")

# --- DATASET 4: Reference signal (for PSD demodulation) ---
sf.write("datasets/dataset4_reference.wav", carrier, fs)
print("✅ Saved dataset4_reference.wav")

# --- Visualization ---
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.title("Dataset 1: Baseband signal (40 Hz with 1 Hz amplitude modulation)")
plt.plot(t[:], signal[:])
plt.subplot(3, 1, 2)
plt.title("Dataset 3: AM signal (5 kHz carrier * baseband signal)")
plt.plot(t[:], am_signal[:])
plt.subplot(3, 1, 3)
plt.title("Carrier (5 kHz reference)")
plt.plot(t[:], carrier[:])
plt.tight_layout()
plt.show()