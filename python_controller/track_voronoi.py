#import matplotlib
#matplotlib.use('TkAgg')

import numpy as np

import track

trk = track.load_csv("3yp_track2500")
# Scale the track (its current design is for full sized cars)
trk = track.Track(matrix=trk.matrix * 0.04)

splines = track.parameterise(trk)

# TODO: Obvs move this to a script
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nx = 1000
ny = 1000
x = np.linspace(trk.x().min(), trk.x().max(), nx)
y = np.linspace(trk.y().min(), trk.y().max(), ny)
xvg, yvg = np.meshgrid(x, y)

thetav_spline = splines.theta_spline(x, y, grid=True)
distances_spline = splines.a_spline(x, y, grid=True)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colours = cm.cubehelix(thetav_spline.reshape(nx, ny) / thetav_spline.max())
ax.plot_surface(xvg, yvg, distances_spline.reshape(nx, ny), facecolors=colours)
plt.show()