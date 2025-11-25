import numpy as np


# =========================================================
# 💠 Convert complex shape → Nx2 array and back
# =========================================================

def complex_to_2d(shape):
    """Convert complex vector (N,) to Nx2 real array."""
    return np.column_stack((shape.real, shape.imag))


def to_complex(arr2d):
    """Convert Nx2 real to complex (N,) points."""
    return arr2d[:, 0] + 1j * arr2d[:, 1]


# =========================================================
# 💠 Similarity Procrustes: scale + rotation + translation
# (same logic as your previous assignment)
# =========================================================

def similarity_transform(X_cpx, Y_cpx, eps=1e-8):
    """
    Align Y to X by similarity transform (rotation, translation, scale).
    Stable version that prevents numerical explosion.
    """

    # Convert to real Nx2 arrays
    X = complex_to_2d(X_cpx)
    Y = complex_to_2d(Y_cpx)

    # Means
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)

    # Center
    Xc = X - muX
    Yc = Y - muY

    # Variance of Y
    varY = np.sum(np.sum(Yc**2, axis=1))

    if varY < eps:
        # Degenerate shape: return translation only
        R = np.eye(2)
        s = 1.0
        t = muX - muY
        Y_aligned = Y + t
        return to_complex(Y_aligned), s, R, t

    # Cross-covariance
    H = Xc.T @ Yc

    # SVD for rotation
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt

    # Fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # SCALE — stable formula
    s = np.trace(R.T @ H) / varY

    # Clamp scale to avoid explosion
    if not np.isfinite(s) or s > 10 or s < 0.1:
        s = 1.0

    # Translation
    t = muX - s * (R @ muY)

    # Transform Y
    Y_aligned = (s * (R @ Y.T)).T + t

    return to_complex(Y_aligned), s, R, t




# =========================================================
# 💠 Generalized Procrustes (GPA)
# Follows the lecture EXACTLY (steps 1 → 6)
# =========================================================

def generalized_procrustes(shapes, max_iter=50, tol=1e-6):
    """
    shapes: (N_points, N_shapes) complex matrix
    Returns: aligned_shapes, mean_shape
    """

    # 1. Choose first shape as initial reference
    reference = shapes[:, 0].copy()

    # 2. Align all shapes to the reference once
    aligned = np.zeros_like(shapes, dtype=complex)
    for i in range(shapes.shape[1]):
        aligned[:, i], _, _, _ = similarity_transform(reference, shapes[:, i])

    # 3. Compute initial mean (normalize shape size)
    mean_shape = np.mean(aligned, axis=1)
    mean_shape /= np.linalg.norm(mean_shape)

    for iteration in range(max_iter):

        prev_mean = mean_shape.copy()

        # 4. Align all shapes to current mean
        for i in range(shapes.shape[1]):
            aligned[:, i], _, _, _ = similarity_transform(mean_shape, aligned[:, i])

        # 5. Update mean
        mean_shape = np.mean(aligned, axis=1)

        # 6. Normalize mean to avoid scale drift
        mean_norm = np.linalg.norm(mean_shape)
        if mean_norm < 1e-12:
            raise ValueError("Mean shape collapsed (norm=0).")
        mean_shape /= mean_norm

        # 7. Check convergence
        if np.linalg.norm(mean_shape - prev_mean) < tol:
            print(f"GPA converged in {iteration+1} iterations.")
            break

    return aligned, mean_shape
