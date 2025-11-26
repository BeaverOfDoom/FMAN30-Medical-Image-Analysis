import numpy as np
from skimage import filters, measure, morphology

from .procrustes import similarity_transform
from .shape_model import complex_to_vec, vec_to_complex
from .resampling import resample_landmarks


# ---------------------------------------------------------
# 1. Initial right-kidney mask (very simple)
# ---------------------------------------------------------

def initial_right_kidney_mask(image):
    """
    Very simple initial segmentation:
      - normalize
      - Otsu threshold
      - remove small blobs
      - pick the largest region on the RIGHT half (right kidney)
    """
    img = image.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Global threshold
    th = filters.threshold_otsu(img)
    mask = img > th

    # Clean mask
    mask = morphology.remove_small_objects(mask, min_size=100)
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = morphology.binary_opening(mask, morphology.disk(1))

    # Label components
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)

    if not props:
        return mask  # empty/fallback

    h, w = img.shape
    # components whose centroid is on right half of the image
    right_props = [p for p in props if p.centroid[1] > w / 2]

    if right_props:
        best = max(right_props, key=lambda p: p.area)
    else:
        # fallback: largest component overall
        best = max(props, key=lambda p: p.area)

    kidney_mask = (labeled == best.label)
    return kidney_mask


# ---------------------------------------------------------
# 2. Boundary → resampled 14-point landmarks
# ---------------------------------------------------------

def extract_boundary_points(mask):
    """
    Extract boundary points (x,y) from a binary kidney mask.
    Returns array of shape (M, 2) with columns [x, y].
    """
    contours = measure.find_contours(mask.astype(float), level=0.5)
    if not contours:
        raise RuntimeError("No contour found in initial segmentation.")

    # Take the longest contour
    contour = max(contours, key=lambda c: c.shape[0])  # (row, col)

    # Convert to (x,y) = (col,row)
    boundary_xy = np.column_stack((contour[:, 1], contour[:, 0]))
    return boundary_xy


def boundary_to_14_landmarks(boundary_xy, n_points=14):
    """
    Take dense boundary points (x,y), resample to n_points and
    ensure the first landmark is at the BOTTOM of the kidney
    (largest y).
    Returns complex vector of length n_points.
    """
    pts_cpx = boundary_xy[:, 0] + 1j * boundary_xy[:, 1]  # x + i y

    resampled = resample_landmarks(pts_cpx, n_points=n_points)
    resampled = np.asarray(resampled).ravel()  # (n_points,)

    # First landmark at bottom (max y)
    ys = resampled.imag
    bottom_idx = np.argmax(ys)
    resampled = np.roll(resampled, -bottom_idx)

    return resampled


# ---------------------------------------------------------
# 3. ASM on a single image
# ---------------------------------------------------------

def asm_segment_single_image(
    image,
    mean_shape_cpx,
    mean_vec,
    eigvecs,
    eigvals,
    n_modes=5,
    max_iters=20,
    conv_thresh=0.1,
    verbose=False,
):
    """
    ASM-style segmentation of the RIGHT kidney in one DMSA image.

    Inputs:
      image          : 2D numpy array
      mean_shape_cpx : complex (N,) mean shape in model coords
      mean_vec       : (2N,) mean vector used in PCA
      eigvecs        : (2N, 2N) PCA eigenvectors
      eigvals        : (2N,) eigenvalues
      n_modes        : number of modes to use (t)
      max_iters      : max ASM iterations
      conv_thresh    : stop if mean landmark movement < this

    Returns:
      landmarks      : final landmarks in IMAGE coordinates (complex, length N)
      init_mask      : initial binary mask
      boundary_xy    : dense boundary points (x,y) from initial mask
    """

    # 1) Initial rough segmentation
    init_mask = initial_right_kidney_mask(image)
    boundary_xy = extract_boundary_points(init_mask)

    # 2) Initial 14-point landmarks on that boundary
    N_points = len(mean_shape_cpx)
    landmarks = boundary_to_14_landmarks(boundary_xy, n_points=N_points)

    # Precompute boundary (for nearest-point search)
    B = boundary_xy  # (M,2), columns x,y

    # Restrict PCA basis to first n_modes
    P = eigvecs[:, :n_modes]
    lambdas = eigvals[:n_modes]

    for it in range(max_iters):
        old_landmarks = landmarks.copy()

        # 3) Align landmarks (image coords) to mean shape (model coords)
        aligned_cpx, s, R, t = similarity_transform(mean_shape_cpx, landmarks)
        x_vec = complex_to_vec(aligned_cpx)  # (2N,)

        # 4) Compute shape parameters b in PCA space
        diff = x_vec - mean_vec
        b = P.T @ diff  # (n_modes,)

        # 5) Clamp b to ±3 sqrt(lambda)
        for i in range(n_modes):
            limit = 3 * np.sqrt(lambdas[i])
            b[i] = np.clip(b[i], -limit, limit)

        # 6) Reconstruct constrained shape in MODEL coordinates
        x_new_vec = mean_vec + P @ b
        recon_cpx = vec_to_complex(x_new_vec)

        # 7) Transform back to IMAGE coordinates (inverse transform)
        Z = np.column_stack((recon_cpx.real, recon_cpx.imag))  # (N,2)
        # Forward used: X_model ≈ s * R @ Y_image + t
        # So inverse:   Y_image ≈ (1/s) * R^T @ (X_model - t)
        Y = (1.0 / s) * (R.T @ (Z.T - t.reshape(2, 1))).T  # (N,2)
        landmarks_pred = Y[:, 0] + 1j * Y[:, 1]

        # 8) Snap each landmark to nearest boundary point
        new_landmarks = []
        for z in landmarks_pred:
            zx, zy = z.real, z.imag
            d2 = (B[:, 0] - zx) ** 2 + (B[:, 1] - zy) ** 2
            idx = np.argmin(d2)
            new_landmarks.append(B[idx, 0] + 1j * B[idx, 1])
        landmarks = np.array(new_landmarks)

        # 9) Check convergence
        mean_disp = np.mean(np.abs(landmarks - old_landmarks))
        if verbose:
            print(f"Iter {it+1}: mean landmark displacement = {mean_disp:.3f} px")
        if mean_disp < conv_thresh:
            break

    return landmarks, init_mask, boundary_xy
