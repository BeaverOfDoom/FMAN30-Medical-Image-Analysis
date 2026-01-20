import numpy as np

def resample_landmarks(points, n_points=14):
    pts = np.asarray(points).reshape(-1)
    
    # Distances between points 
    distances = np.abs(np.diff(pts, append=pts[0]))

    # Arc length should have same length as pts
    arc_length = np.concatenate([[0], np.cumsum(distances[:-1])])
    arc_length /= arc_length[-1]  # normalize

    # Target positions
    target = np.linspace(0, 1, n_points, endpoint=False)

    # Interpolate complex points
    resampled = np.interp(target, arc_length, pts)

    return resampled