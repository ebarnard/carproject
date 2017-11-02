from math import hypot
import numpy as np
import os
import scipy as sp
from scipy.interpolate import RectBivariateSpline, CubicSpline
from scipy.spatial import cKDTree
import typing

#TODO: Arc length parameterisation - https://mathoverflow.net/questions/13793/finding-the-length-of-a-cubic-b-spline

class Track(typing.NamedTuple):
    matrix: np.ndarray

    def x(self):
        return self.matrix[0,:]

    def y(self):
        return self.matrix[1,:]

    def xy(self):
        return self.matrix[(0,1),:]

    def dx(self):
        return self.matrix[2,:]

    def dy(self):
        return self.matrix[3,:]

    def dxdy(self):
        return self.matrix[(2,3),:]

    def n(self):
        return self.matrix.shape[1]

def load_csv(track: str):
    path = os.path.join(os.path.dirname(__file__), "data", "tracks", track + ".csv")
    matrix = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1).T
    return Track(matrix = matrix)

class CurvilinearMapping(typing.NamedTuple):
    track_kd: cKDTree
    distances: np.array
    theta_max: float
    theta_spline: RectBivariateSpline
    theta_spline_shifted: RectBivariateSpline
    a_spline: RectBivariateSpline

    xy_spline: CubicSpline

    # TODO: The 2D spline isn't working yet so use this instead
    def theta(self, x, y):
        _, indices = self.track_kd.query([x, y])
        return self.distances[indices]

    def theta_continuous(self, x, y, dx=0, dy=0):
        # Try to approximate constant time operation
        thetas = self.theta_spline(x, y, grid=False)
        use_shifted = thetas < 0 and thetas > 2

        vals = self.theta_spline(x, y, dx=dx, dy=dy, grid=False)
        vals_shifted = self.theta_spline_shifted(x, y, dx=dx, dy=dy, grid=False)

        return np.where(use_shifted, vals_shifted, vals)

    def a(self, x, y, dx=0, dy=0):
        return self.a_spline(x, y, dx=dx, dy=dy, grid=False)

    def xy(self, theta, dtheta=0):
        return self.xy_spline(theta, nu=dtheta)

def parameterise(trk: Track):
    # Calculate track spline distances
    distance_to_next = trk.xy() - np.roll(trk.xy(), -1, axis=1)
    distance_to_next = np.hypot(distance_to_next[0,:], distance_to_next[1,:])
    distance = np.concatenate(([0], np.cumsum(distance_to_next)))

    # Create KD-tree of track line positions
    track_kd = cKDTree(trk.xy().T)

    # Create matrix of grid positions
    nx = 1000
    ny = 1000
    x = np.linspace(trk.x().min(), trk.x().max(), nx)
    y = np.linspace(trk.y().min(), trk.y().max(), ny)
    xvg, yvg = np.meshgrid(x, y)
    xv, yv = np.ndarray.flatten(xvg), np.ndarray.flatten(yvg)
    grid = np.array((xv, yv))

    # Calculate nearest track point and distance for each grid point
    distances, indices = track_kd.query(grid.T)

    # Matrix grid containing nearest theta value against x, y
    theta_grid = distance[indices].reshape(nx, ny)

    # For each intersection theta=track index point, a = distance from track index point
    # Create bivariate spline mapping x, y to theta, a
    theta_spline = RectBivariateSpline(x, y, theta_grid)
    # Theta is a periodic variable and therefore there is a sharp discontinutiy crossing zero
    # in its spline approximation. To avoid this discontinuity a second spline approximation 
    # is created starting from half way around the track.
    # TODO: Add theta_max to the first half of the indices
    theta_spline_shifted = RectBivariateSpline(x, y, theta_grid)
    a_spline = RectBivariateSpline(x, y, distances.reshape(nx, ny))

    xy_overlapped = np.concatenate((trk.xy(), trk.xy()[:,0:1]), axis=1)
    xy_spline = CubicSpline(distance, xy_overlapped, axis=1, bc_type='periodic')

    return CurvilinearMapping(
        track_kd=track_kd,
        theta_max=distance[-1],
        theta_spline=theta_spline,
        theta_spline_shifted=theta_spline_shifted,
        a_spline=a_spline,
        xy_spline=xy_spline,
        distances=distance
    )

