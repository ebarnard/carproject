import cv2
import numpy as np

def create_mask(track_centreline_csv, H, width, height):
    track_centreline = np.genfromtxt(track_centreline_csv, dtype=None, delimiter=',', skip_header=1)

    centreline_gradient = (np.roll(track_centreline, -3) - np.roll(track_centreline, 3)) / 2

    # normalise gradient vector to be unit length
    length = np.hypot(centreline_gradient[:, 0], centreline_gradient[:, 1])
    centreline_gradient[:, 0] /= length
    centreline_gradient[:, 1] /= length

    # rotate gradient vector by pi to find the normal vector
    centreline_normal = np.array([-centreline_gradient[:, 1], centreline_gradient[:, 0]])

    # inner and outer may be swapped but that doesn't matter
    track_inner = np.array([track_centreline[:, 0] + centreline_normal[0, :] * (track_centreline[:, 2]/2),
                            track_centreline[:, 1] + centreline_normal[1, :] * (track_centreline[:, 2]/2)])
    track_outer = np.array([track_centreline[:, 0] - centreline_normal[0, :] * (track_centreline[:, 2]/2),
                            track_centreline[:, 1] - centreline_normal[1, :] * (track_centreline[:, 2]/2)])

    # project the track bounds into image coordinates
    track_inner_proj = H.dot(np.r_[track_inner, [np.ones(2500)]])
    track_outer_proj = H.dot(np.r_[track_outer, [np.ones(2500)]])

    # plot the projected bounds
    #import matplotlib.pyplot as plt
    #plt.plot(track_inner_proj[0, :], track_inner_proj[1, :], "g")
    #plt.plot(track_outer_proj[0, :], track_outer_proj[1, :], "b")
    #plt.axis("equal")

    img = np.zeros((height, width), np.uint8)

    # fill in the inner and outer track polygons
    # regions that overlap an even number of times are not drawn
    cv2.fillPoly(img, np.array([track_outer_proj[0:2, :].transpose(), track_inner_proj[0:2, :].transpose()], np.int32), 255)

    return img
