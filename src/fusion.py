import numpy as np

def fuse_measurements(radar_meas, video_meas, w_radar=0.6):
    return w_radar * radar_meas + (1 - w_radar) * video_meas
