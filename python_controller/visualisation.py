import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

from runner import SimHistory

def show(trk, dt: float, history: SimHistory):
    n = history.n
    t = np.arange(n) * dt
    v = np.hypot(history.velocities[0,:], history.velocities[1,:])

    # Setup output plot
    fig = plt.figure()
    position_ax = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
    velocity_ax = plt.subplot2grid((2, 4), (0, 2))
    slip_angle_ax = plt.subplot2grid((2, 4), (1, 2))
    steering_angle_ax = plt.subplot2grid((2, 4), (1, 3))

    # Plot track extremities
    position_ax.plot(trk.x(), trk.y())

    # Plot potision on track
    position_ax.add_collection(speed_coloured_linecollection(history.positions, v))
    position_ax.axis("equal")

    # Plot velocities
    tv = np.stack((t, v));
    velocity_ax.add_collection(speed_coloured_linecollection(tv, v))
    velocity_ax.axis("auto")
    velocity_ax.set_title("Velocity")
    velocity_ax.set_xlabel("t (s)")
    velocity_ax.set_ylabel("v (m/s)")

    # Plot slip angles
    slip_angles = np.unwrap(history.headings - np.arctan2(history.velocities[1,:], history.velocities[0,:]))
    slip_angle_ax.plot(t, slip_angles[0,:])
    slip_angle_ax.set_title("Slip Angle")
    slip_angle_ax.set_xlabel("t (s)")
    slip_angle_ax.set_ylabel("theta (rad)")

    # Plot steering angles
    steering_angle_ax.plot(t, np.unwrap(history.steering_angles[0,:]))
    steering_angle_ax.set_title("Steering Angle")
    steering_angle_ax.set_xlabel("t (s)")
    steering_angle_ax.set_ylabel("theta (rad)")

    #ani = CarAnimaiton(position_ax, history, dt)

    plt.tight_layout()
    plt.show(block=True)

def speed_coloured_linecollection(xy, v) -> mpl.collections.LineCollection:
    segments = xy_to_linesegments(xy)
    norm = mpl.colors.Normalize(0, v.max())
    lc = mpl.collections.LineCollection(segments, cmap='plasma', norm=norm)
    lc.set_array(v)
    lc.set_linewidth(3)
    return lc

def xy_to_linesegments(xy) -> mpl.collections.LineCollection:
    points = xy.T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)

class CarAnimaiton(mpl.animation.TimedAnimation):
    def __init__(self, axes: mpl.axes.Axes, history: SimHistory, dt: float):
        self.history = history
        self.line = mpl.lines.Line2D([], [], color='black')
        axes.add_line(self.line)

        mpl.animation.TimedAnimation.__init__(self, axes.get_figure(), interval=0.01, blit=True)

    def _draw_frame(self, framedata):
        car = np.matrix([[-0.05, 0.05, 0.05, -0.05, -0.05],
                         [-0.02, -0.02, 0.02, 0.02, -0.02]])

        i = framedata
        heading = self.history.headings[0,i]
        c, s = np.cos(heading), np.sin(heading)
        R = np.matrix([[c, -s], [s, c]])
        car = R * car
        x,y = self.history.positions[:,i]

        self.line.set_data(car[0,:] + x, car[1,:] + y)
        self._drawn_artists = [self.line]

    def new_frame_seq(self):
        return iter(range(self.history.n))

    def _init_draw(self):
        self.line.set_data([], [])
