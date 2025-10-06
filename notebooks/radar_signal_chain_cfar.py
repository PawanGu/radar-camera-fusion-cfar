import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft2

# -----------------------------
# Synthetic Radar Parameters
# -----------------------------
num_chirps   = 64      # Doppler (slow-time) bins
num_samples  = 128     # Range (fast-time) bins
noise_level  = 0.3

# Two targets: (range_bin, doppler_bin, amplitude)
targets = [
    (40,  10, 1.0),
    (95, -12, 0.9),
]

# -----------------------------
# Generate Synthetic Signal
# -----------------------------
signal = np.zeros((num_chirps, num_samples), dtype=complex)
t_fast = np.arange(num_samples)
t_slow = np.arange(num_chirps)[:, None]

for rng, dop, amp in targets:
    range_tone   = np.exp(1j * 2*np.pi * rng * t_fast / num_samples)
    doppler_tone = np.exp(1j * 2*np.pi * dop * t_slow / num_chirps)
    signal += amp * doppler_tone * range_tone

# Window along fast-time (range) to reduce sidelobes
window = np.hanning(num_samples)
signal *= window

# Add complex Gaussian noise
signal += noise_level * (np.random.randn(num_chirps, num_samples) + 1j*np.random.randn(num_chirps, num_samples))

# -----------------------------
# Range–Doppler Map (linear power)
# -----------------------------
rd = np.abs(fftshift(fft2(signal)))**2
rd /= rd.max() + 1e-12  # normalize for plotting

# -----------------------------
# CA-CFAR (2D) Implementation
# -----------------------------
def ca_cfar_2d(power_map, guard=1, train=4, pfa=1e-3):
    """
    2D Cell-Averaging CFAR on linear power map.
    guard: guard cells on each side (square guard region)
    train: training cells thickness around guard
    pfa: desired probability of false alarm
    Returns boolean detection map of same shape.
    """
    H, W = power_map.shape
    half = guard + train  # half window radius including guard+train

    # Integral image for fast rectangular sums
    I = np.pad(power_map, ((1,0),(1,0)), mode='constant')
    I = np.cumsum(np.cumsum(I, axis=0), axis=1)

    def rect_sum(y0, x0, y1, x1):
        return I[y1+1, x1+1] - I[y0, x1+1] - I[y1+1, x0] + I[y0, x0]

    det = np.zeros_like(power_map, dtype=bool)
    N_train = (2*half+1)**2 - (2*guard+1)**2
    alpha = N_train * (pfa**(-1.0 / max(N_train,1)) - 1.0)

    for y in range(half, H-half):
        for x in range(half, W-half):
            y0, x0 = y-half, x-half
            y1, x1 = y+half, x+half
            total_sum = rect_sum(y0, x0, y1, x1)

            yg0, xg0 = y-guard, x-guard
            yg1, xg1 = y+guard, x+guard
            guard_sum = rect_sum(yg0, xg0, yg1, xg1)

            noise_sum = total_sum - guard_sum
            noise_mean = noise_sum / max(N_train,1)
            threshold = alpha * noise_mean

            if power_map[y, x] > threshold:
                det[y, x] = True
    return det

# Run CFAR
detections = ca_cfar_2d(rd, guard=1, train=4, pfa=1e-4)

# -----------------------------
# Visualization
# -----------------------------
rd_db = 10*np.log10(rd + 1e-12)

plt.figure(figsize=(7,5))
plt.imshow(rd_db, aspect='auto',
           extent=[0, num_samples, -num_chirps//2, num_chirps//2],
           cmap='viridis')
plt.xlabel('Range bin')
plt.ylabel('Doppler bin')
plt.title('Range–Doppler Map with 2D CA-CFAR Detections (dB)')
cbar = plt.colorbar()
cbar.set_label('Power [dB]')

# Overlay detections as red squares
ys, xs = np.where(detections)
doppler_vals = ys - (num_chirps // 2)
plt.scatter(xs, doppler_vals, marker='s', s=24, facecolors='none', edgecolors='red', linewidths=1.2, label='CFAR detections')

plt.legend(loc='upper right')
plt.tight_layout()

# Save figure (add this line)
plt.savefig("range_doppler_map_cfar.png", dpi=300, bbox_inches='tight')

plt.show()
