'''Set of helper functions for matrix operations'''
import numpy as np


def calculate_dist(pts1, pts2):
    '''Calculate the distance between two sets of N-D vectors. Assumes pts is of size M x N. I.e M N-D vectors'''
    diff = pts1 - pts2
    return np.linalg.norm(diff, axis=1)

def apply_matrix(pts, F):
    ''' 
    apply matrix F to the input points, first converting to homogenous
    Params:
    pts: (Nxm) matrix of points
    F: (m+1 x m+1) matrix (could be fundmental)
    '''  
    # Convert to homgenous coords  
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1)))) # (N x m)

    # Transform by matrix F
    transformed_pts = (F @ pts_hom.T).T  # (N x m)

    # Divide by homogenous scaling
    return transformed_pts[:, 0:pts.shape[1]] / transformed_pts[:, pts.shape[1]][:, None]