import numpy as np
from simulate_radar import simulate_radar_targets
from simulate_video import simulate_video_from_radar
from kalman_filter import KalmanFilterCV
from data_association import assign_tracks
from fusion import fuse_measurements
from visualize import init_plot, update_plot

def main():
    num_targets = 3
    num_frames = 80
    radar_data = simulate_radar_targets(num_targets, num_frames)
    video_data = simulate_video_from_radar(radar_data)

    kfs = [KalmanFilterCV() for _ in range(num_targets)]
    for kf, pos in zip(kfs, radar_data[0]):
        kf.init_state(pos)
    tracks = [[] for _ in range(num_targets)]

    fig, ax = init_plot()

    for t in range(num_frames):
        radar_meas = radar_data[t]
        video_meas = video_data[t]
        fused = fuse_measurements(radar_meas, video_meas)

        preds = np.array([kf.predict() for kf in kfs])
        pairs = assign_tracks(fused, preds)

        for i, j in pairs:
            est = kfs[i].update(fused[j])
            tracks[i].append(est.tolist())

        update_plot(ax, radar_meas, video_meas, fused, tracks)

    print("Demo complete. Close window to exit.")
    input("Press Enter to close...")

if __name__ == "__main__":
    main()
