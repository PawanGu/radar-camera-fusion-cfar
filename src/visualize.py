import matplotlib.pyplot as plt

def init_plot():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Radar–Camera Fusion Tracker")
    ax.legend()
    return fig, ax

def update_plot(ax, radar_meas, video_meas, fused_meas, tracks):
    ax.clear()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.plot(radar_meas[:,0], radar_meas[:,1], 'bo', label='Radar')
    ax.plot(video_meas[:,0], video_meas[:,1], 'ro', label='Video')
    ax.plot(fused_meas[:,0], fused_meas[:,1], 'go', label='Fused')
    for tr in tracks:
        xs = [p[0] for p in tr]
        ys = [p[1] for p in tr]
        ax.plot(xs, ys, 'k-', alpha=0.6)
    ax.legend()
    plt.pause(0.05)
