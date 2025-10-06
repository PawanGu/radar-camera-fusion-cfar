import numpy as np

def simulate_video_from_radar(radar_data, offset=(0.4, -0.3), jitter=0.15):
    np.random.seed(1)
    offset = np.array(offset)
    video_data = radar_data + offset + np.random.randn(*radar_data.shape) * jitter
    return video_data
