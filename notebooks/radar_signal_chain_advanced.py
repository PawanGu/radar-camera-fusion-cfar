import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft2

# Parameters
num_chirps = 64
num_samples = 128
noise_level = 0.3

# Define multiple targets (range_bin, doppler_bin, amplitude)
targets = [
    (40, 10, 1.0),   # Target 1
    (80, -15, 0.8)   # Target 2
]

# Build synthetic beat signal
signal = np.zeros((num_chirps, num_samples), dtype=complex)
t = np.arange(num_samples)
doppler_axis = np.arange(num_chirps)

for rng, dop, amp in targets:
    tone = amp * np.exp(1j * 2*np.pi * rng * t / num_samples)
    doppler_tone = np.exp(1j * 2*np.pi * dop * doppler_axis[:, None] / num_chirps)
    signal += tone * doppler_tone

# Add windowing to reduce sidelobes
window = np.hanning(num_samples)
signal *= window

# Add complex Gaussian noise
signal += noise_level * (np.random.randn(num_chirps, num_samples) + 1j*np.random.randn(num_chirps, num_samples))

# Compute 2D FFT for Range–Doppler map
range_doppler = np.abs(fftshift(fft2(signal)))**2
range_doppler_db = 10 * np.log10(range_doppler / np.max(range_doppler) + 1e-6)

# Plot
plt.figure(figsize=(6,4))
plt.imshow(range_doppler_db, aspect='auto',
           extent=[0, num_samples, -num_chirps/2, num_chirps/2],
           cmap='viridis')
plt.xlabel('Range bin')
plt.ylabel('Doppler bin')
plt.title('Synthetic Range–Doppler Map (2 Targets, dB Scale)')
plt.colorbar(label='Power [dB]')
plt.tight_layout()

# Save figure (add this line)
plt.savefig("range_doppler_map.png", dpi=300, bbox_inches='tight')

# Show figure (optional)
plt.show()

