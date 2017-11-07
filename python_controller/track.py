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
    segment_distances: np.array
    xy_spline: CubicSpline

    def theta(self, x, y):
        _, indices = self.track_kd.query([x, y])
        return self.segment_distances[indices]

    def xy(self, theta, dtheta=0):
        return self.xy_spline(theta, nu=dtheta)

def parameterise(trk: Track):
    # Calculate track spline distances
    distance_to_next = trk.xy() - np.roll(trk.xy(), -1, axis=1)
    distance_to_next = np.hypot(distance_to_next[0,:], distance_to_next[1,:])
    segment_distances = np.concatenate(([0], np.cumsum(distance_to_next)))

    # Create KD-tree of track line positions
    track_kd = cKDTree(trk.xy().T)

    xy_overlapped = np.concatenate((trk.xy(), trk.xy()[:,0:1]), axis=1)
    xy_spline = CubicSpline(segment_distances, xy_overlapped, axis=1, bc_type='periodic')

    return CurvilinearMapping(
        track_kd=track_kd,
        xy_spline=xy_spline,
        segment_distances=segment_distances
    )

