import sounddevice as sd
import soundfile as sf
import time

# --- PARAMETERS ---
fs = 44100
datasets = [
    "dataset1_base_signal.wav",
    "dataset2_signal_noisy.wav",
    "dataset3_am_noisy.wav"
]

# --- Loop through all datasets ---
for filename in datasets:
    print(f"\n▶️ Playing and recording {filename}...")
    data, fs = sf.read(filename)

    # Normalize just in case
    data = data / max(abs(data))

    print("Get ready... (3s)")
    time.sleep(3)

    print("Recording...")
    recording = sd.playrec(data, samplerate=fs, channels=1, dtype='float32', blocking=True)
    print("✅ Done!")

    # Save recorded mic input
    outname = filename.replace(".wav", "_recorded.wav")
    sf.write(outname, recording, fs)
    print(f"💾 Saved: {outname}")

print("\n🎉 All recordings complete!")