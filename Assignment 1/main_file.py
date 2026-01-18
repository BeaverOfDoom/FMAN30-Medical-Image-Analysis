import numpy as np
import cv2
from skimage.feature import match_descriptors, plot_matched_features, SIFT
import matplotlib.pyplot as plt
from pathlib import Path




# SIFT Feature Extraction and Matching
def sift_extractor(img, extractor):
    extractor.detect_and_extract(img)
    return extractor.keypoints, extractor.descriptors

def mach_features(img1, img2, extractor, num=0):
    kp1, desc1 = sift_extractor(img1, extractor)
    kp2, desc2 = sift_extractor(img2, extractor)
    matches = match_descriptors(desc1, desc2, max_ratio=0.9, cross_check=True)
    print(f"Pair {num+1}: {len(matches)} raw matches found.")

    return matches, kp1, kp2


# Procrustes Alignment (Rigid + Similarity)

def rigid_transformation(X, Y):
    X_mean, Y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
    Xc, Yc = X - X_mean, Y - Y_mean
    H = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Y_mean - R @ X_mean
    return R, t # Basically we just estimate rotation and translation

def similarity_transformation(X, Y):
    X_mean, Y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)
    Xc, Yc = X - X_mean, Y - Y_mean
    H = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_X = np.sum(np.linalg.norm(Xc, axis=1)**2)
    if var_X < 1e-8: # To not divide by zero
        s = 1.0
    else:
        s = np.sum(S) / var_X
        
    t = Y_mean - s * R @ X_mean
    return s, R, t # Basically we just estimate scale, rotation and translation


# RANSAC rigid and similarity
def ransac_rigid(X, Y, n_iters=5000, threshold=3.0):
    
    # Initializing variables
    N = X.shape[0]
    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0
    best_R, best_t = None, None

    # RANSAC iterations
    for _ in range(n_iters):
        idx = np.random.permutation(N)[:2] # Select 2 random points
        R, t = rigid_transformation(X[idx], Y[idx]) # Estimate transformation
        Y_pred = (R @ X.T).T + t # Apply transformation
        errors = np.linalg.norm(Y - Y_pred, axis=1) # Compute errors
        inliers = errors < threshold # Determine inliers
        count = np.sum(inliers) # Count inliers
        
        # Update best model if current one is better
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_R, best_t = R, t

    if best_count >= 2: # Recompute using all inliers
        best_R, best_t = rigid_transformation(X[best_inliers], Y[best_inliers])

    return best_R, best_t, best_inliers # Return best transformation and inliers


def ransac_similarity(X, Y, n_iters=5000, threshold=5.0):
    
    # Initializing variables
    N = X.shape[0]
    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0
    best_s, best_R, best_t = 1, np.eye(2), np.zeros(2)

    # RANSAC iterations
    for _ in range(n_iters):
        
        idx = np.random.choice(N, 2, replace=False) # Select 2 random points
        s, R, t = similarity_transformation(X[idx], Y[idx]) # Estimate transformation
        Y_pred = (s * (R @ X.T)).T + t # Apply transformation
        errors = np.linalg.norm(Y - Y_pred, axis=1) # Compute errors
        inliers = errors < threshold # Determine inliers
        count = np.sum(inliers) # Count inliers
        
        # Update best model if current one is better
        if count > best_count:
            best_count = count
            best_inliers, best_s, best_R, best_t = inliers, s, R, t

    if best_count >= 2: 
        best_s, best_R, best_t = similarity_transformation(X[best_inliers], Y[best_inliers])

    return best_s, best_R, best_t, best_inliers # Return best transformation and inliers


# Just for visualization
def show_alignment(img1, img2, aligned, title_prefix="", overlay=False):

    if overlay:
        # overlap view
        blended = cv2.addWeighted(img2, 0.5, aligned, 0.5, 0)
        plt.figure(figsize=(7, 7))
        plt.imshow(blended, cmap='gray')
        plt.title(f"{title_prefix}: Overlay (Aligned + Target)") # Here we change the title with the title_prefix
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    else:
        # Side by side view
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        axs[0].imshow(aligned, cmap='gray')
        axs[0].set_title(f"{title_prefix}: Aligned (Transformed)")
        axs[1].imshow(img2, cmap='gray')
        axs[1].set_title(f"{title_prefix}: Target")
        for a in axs:
            a.axis('off')
        plt.tight_layout()
        plt.show()
    

# Preprocessing
def preProcess_HE_AMACR(he_dir, amacr_dir):
    
    he_dir, amacr_dir = Path(he_dir), Path(amacr_dir)
    patterns = ["*.bmp", "*.jpg", "*.jpeg", "*.tif", "*.png"]

    def get_paths(folder):
        paths = []
        for p in patterns:
            paths.extend(folder.glob(p))
        return sorted(paths)

    he_paths, amacr_paths = get_paths(he_dir), get_paths(amacr_dir)
    print(f"Found {len(he_paths)} HE and {len(amacr_paths)} AMACR images.")

    def process(paths):
        imgs = []
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
            h, w = gray.shape 
            gray = cv2.resize(gray, (w // 2, h // 2)) # Resize to half
            imgs.append(gray)
        return imgs

    return process(he_paths), process(amacr_paths)


def preProcess_HE_TRF(he_dir, trf_dir):
    he_dir, trf_dir = Path(he_dir), Path(trf_dir)
    patterns = ["*.bmp", "*.jpg", "*.jpeg", "*.tif", "*.png"]

    def get_paths(folder):
        paths = []
        for p in patterns:
            paths.extend(folder.glob(p))
        return sorted(paths)

    he_paths, trf_paths = get_paths(he_dir), get_paths(trf_dir)
    print(f"Found {len(he_paths)} HE and {len(trf_paths)} TRF images.")

    def process(paths, invert=False):
        imgs = []
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if invert:
                gray = cv2.bitwise_not(gray)  # Invert colors for TRF images
            gray = cv2.equalizeHist(gray) 
            h, w = gray.shape
            gray = cv2.resize(gray, (w // 2, h // 2)) # Resize to half
            imgs.append(gray)
        return imgs

    return process(he_paths, invert=False), process(trf_paths, invert=True)



# Task 1 — Rigid
def run_collection1(plot_example=True, overlay=False):
    
    # path
    HE_path = r"C:\Users\46734\OneDrive\Dokument\LTH\Kurser\Medical Image Analysis\FMAN30-Medical-Image-Analysis-main\Assignment 1\Collection 1\HE"
    AMACR_path = r"C:\Users\46734\OneDrive\Dokument\LTH\Kurser\Medical Image Analysis\FMAN30-Medical-Image-Analysis-main\Assignment 1\Collection 1\p63AMACR"

    print("HE_path exists:", Path(HE_path).exists())
    print("AMACR_path exists:", Path(AMACR_path).exists())
    print("HE files:", len(list(Path(HE_path).glob("*"))))
    print("AMACR files:", len(list(Path(AMACR_path).glob("*"))))


    # Preprocess images
    images1, images2 = preProcess_HE_AMACR(HE_path, AMACR_path)
    n_pairs = min(len(images1), len(images2))
    print(f"\nProcessing {n_pairs} pairs (Collection 1 — rigid)...\n")

    sift = SIFT()
    results = []
    
    # Main loop over image pairs
    for i in range(n_pairs): 
        gray1, gray2 = images1[i], images2[i]
        matches, kp1, kp2 = mach_features(gray1, gray2, sift, num=i)

        if len(matches) < 6:
            print(f"Pair {i+1}: Too few matches ({len(matches)}), skipping.\n")
            continue

        X = np.float32([[kp1[a][1], kp1[a][0]] for a in matches[:, 0]])
        Y = np.float32([[kp2[b][1], kp2[b][0]] for b in matches[:, 1]])

        R, t, inliers = ransac_rigid(X, Y)
        theta = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        tmag = np.linalg.norm(t)
        results.append((i+1, len(matches), np.sum(inliers), theta, tmag))

        # --- Only visualize first pair for clarity
        if plot_example:
            M = np.hstack([R, t.reshape(2, 1)])
            aligned = cv2.warpAffine(gray1, M, (gray2.shape[1], gray2.shape[0]))

            show_alignment(gray1, gray2, aligned,
                           title_prefix=f"Pair {i+1} (Rigid)",
                           overlay=overlay) 

    # --- Summary Table (outside loop)
    print("\n" + "=" * 55)
    print(f"{'Pair':<6}{'Matches':<10}{'Inliers':<10}{'Rot(°)':<10}{'|t|(px)':<10}")
    print("-" * 55)
    for (pair, total, inl, theta, tmag) in results:
        print(f"{pair:<6}{total:<10}{inl:<10}{theta:<10.2f}{tmag:<10.1f}")
    print("=" * 55)




# Task 2 - Similarity
def run_collection2(plot_example=True, overlay=False):
    HE2_path = r"C:\Users\46734\OneDrive\Dokument\LTH\Kurser\Medical Image Analysis\FMAN30-Medical-Image-Analysis-main\Assignment 1\Collection 2\HE"
    TRF2_path = r"C:\Users\46734\OneDrive\Dokument\LTH\Kurser\Medical Image Analysis\FMAN30-Medical-Image-Analysis-main\Assignment 1\Collection 2\TRF"

    images1, images2 = preProcess_HE_TRF(HE2_path, TRF2_path)
    n_pairs = min(len(images1), len(images2))
    print(f"\nProcessing {n_pairs} pairs (Collection 2 — similarity)...\n")

    sift = SIFT()
    results = []


    for i in range(n_pairs):
        gray1, gray2 = images1[i], images2[i]
        matches, kp1, kp2 = mach_features(gray1, gray2, sift, num=i)
        if len(matches) < 6:
            continue

        X = np.float32([[kp1[a][1], kp1[a][0]] for a in matches[:, 0]])
        Y = np.float32([[kp2[b][1], kp2[b][0]] for b in matches[:, 1]])

        s, R, t, inliers = ransac_similarity(X, Y)
        theta = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        tmag = np.linalg.norm(t)
        results.append((i+1, len(matches), np.sum(inliers), s, theta, tmag))

        if plot_example:
            M = np.hstack([s * R, t.reshape(2, 1)]) if "s" in locals() else np.hstack([R, t.reshape(2, 1)])
            aligned = cv2.warpAffine(gray1, M, (gray2.shape[1], gray2.shape[0]))
            show_alignment(gray1, gray2, aligned,
                        title_prefix=f"Pair {i+1}",
                        overlay=overlay)  


    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Pair':<6}{'Matches':<10}{'Inliers':<10}{'Scale':<10}{'Rotation (deg)':<10}{'t (px)':<10}")
    print("-" * 65)
    for (pair, total, inl, s, theta, tmag) in results:
        print(f"{pair:<6}{total:<10}{inl:<10}{s:<10.3f}{theta:<10.2f}{tmag:<10.1f}")
    print("=" * 65)


if __name__ == "__main__":
    # Here we can change which collection to run.
    # Also, we can toggle if we want plotting or not since it might be annoying sometimes to have to quit out of many plots.
    # Also also, we can toggle overlay or side-by-side visualization. I noticed that it is nice to be able to see both options.
    
    run_collection1(plot_example=False, overlay=False)
    #run_collection2(plot_example=False, overlay=False)
