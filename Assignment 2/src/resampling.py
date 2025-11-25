import numpy as np

def resample_landmarks(points, n_points=14):
    """
    Resample an ordered set of complex contour points to n equally spaced points.
    """
    pts = np.asarray(points).reshape(-1)
    
    # Distances between points (including last→first)
    distances = np.abs(np.diff(pts, append=pts[0]))

    # Arc length should have SAME length as pts
    # So we drop the last cumulative distance (wrap-around)
    arc_length = np.concatenate([[0], np.cumsum(distances[:-1])])
    arc_length /= arc_length[-1]  # normalize

    # Target arc-length positions
    target = np.linspace(0, 1, n_points, endpoint=False)

    # Interpolate along complex numbers
    resampled = np.interp(target, arc_length, pts)

    return resampled