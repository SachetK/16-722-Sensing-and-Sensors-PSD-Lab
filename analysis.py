import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# --- FILES (already generated) ---
original_file = "recorded/dataset1_base_signal_recorded.wav"
am_noisy_file = "recorded/dataset3_am_noisy_recorded.wav"
psd_file = "analysis/dataset4_recovered_psd.wav"
bandpass_file = "analysis/dataset5_recovered_bandpass.wav"

# --- LOAD SIGNALS ---
original_signal, fs = sf.read(original_file)
am_noisy, _ = sf.read(am_noisy_file)
psd_recovered, _ = sf.read(psd_file)
bandpass_recovered, _ = sf.read(bandpass_file)

# Flatten and match lengths
original_signal = original_signal.flatten()
am_noisy = am_noisy.flatten()
psd_recovered = psd_recovered.flatten()
bandpass_recovered = bandpass_recovered.flatten()

# Remove DC offset
original_signal -= np.mean(original_signal)
psd_recovered -= np.mean(psd_recovered)
bandpass_recovered -= np.mean(bandpass_recovered)

n = min(len(original_signal), len(am_noisy), len(psd_recovered), len(bandpass_recovered))
t = np.arange(n) / fs
original_signal = original_signal[:n]
am_noisy = am_noisy[:n]
psd_recovered = psd_recovered[:n]
bandpass_recovered = bandpass_recovered[:n]

# Flatten and match lengths as before
original_signal = original_signal[:n].flatten()
psd_recovered = psd_recovered[:n].flatten()
bandpass_recovered = bandpass_recovered[:n].flatten()

# Normalize all to ±1
original_signal /= np.max(np.abs(original_signal))
psd_recovered /= np.max(np.abs(psd_recovered))
bandpass_recovered /= np.max(np.abs(bandpass_recovered))


# --- METRICS ---
def snr_db(original, recovered):
    noise = original - recovered
    return 10 * np.log10(np.sum(original**2) / np.sum(noise**2))

snr_psd = snr_db(original_signal, psd_recovered)
snr_bp = snr_db(original_signal, bandpass_recovered)

corr_psd = np.corrcoef(original_signal, psd_recovered)[0,1]
corr_bp = np.corrcoef(original_signal, bandpass_recovered)[0,1]

residual_psd = np.sum((original_signal - psd_recovered)**2)
residual_bp = np.sum((original_signal - bandpass_recovered)**2)

print(f"SNR PSD: {snr_psd:.2f} dB, Bandpass: {snr_bp:.2f} dB")
print(f"Correlation PSD: {corr_psd:.3f}, Bandpass: {corr_bp:.3f}")
print(f"Residual Noise PSD: {residual_psd:.5f}, Bandpass: {residual_bp:.5f}")

# --- TIME-DOMAIN PLOTS ---
plt.figure(figsize=(12,10))
plt.subplot(3,1,1)
plt.title("Original Baseband Signal (Dataset 1)")
plt.plot(t[:], original_signal[:])
plt.subplot(3,1,2)
plt.title("PSD Recovered Signal (Dataset 4)")
plt.plot(t[:], psd_recovered[:])
plt.subplot(3,1,3)
plt.title("Frequency-Selective Filter Recovered Signal (Dataset 5)")
plt.plot(t[:], bandpass_recovered[:])
plt.tight_layout()
plt.savefig("time_comparison.pdf")
plt.show()