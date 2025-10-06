import numpy as np

def simulate_radar_targets(num_targets=3, num_frames=100, dt=0.1, noise_std=0.3):
    np.random.seed(0)
    positions = np.random.rand(num_targets, 2) * 10.0
    velocities = (np.random.rand(num_targets, 2) - 0.5) * 2.0
    radar_data = []
    for _ in range(num_frames):
        positions += velocities * dt
        frame = positions + np.random.randn(*positions.shape) * noise_std
        radar_data.append(frame.copy())
    return np.array(radar_data)
