import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Your modules
from src.visualization import show_manual_segmentation, draw_shape
from src.resampling import resample_landmarks
from src.procrustes import generalized_procrustes
from src.shape_model import build_shape_model, plot_eigenvalues, show_modes, complex_to_vec, vec_to_complex
from src.asm_segmentation import asm_segment_single_image



# -------------------------------------------------------
#  PATHS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_data():
    models = loadmat(os.path.join(DATA_DIR, "models.mat"))["models"]     # (14, 40)
    man_seg = loadmat(os.path.join(DATA_DIR, "man_seg.mat"))["man_seg"] # (40, 1)
    images = loadmat(os.path.join(DATA_DIR, "dmsa_images.mat"))["dmsa_images"]  # 128×128×25
    return models, man_seg, images

def choose_num_modes(eigvals, target_energy=0.95):
    """
    Given eigenvalues (1D array), return the smallest t such that
    sum_{i=1..t} eigvals[i] / sum eigvals >= target_energy.
    """
    eigvals = np.asarray(eigvals)
    total = np.sum(eigvals)
    cum = np.cumsum(eigvals)
    ratios = cum / total

    t = int(np.searchsorted(ratios, target_energy) + 1)
    return t, ratios



def main():
    print("------------STARTING PROGRAM------------")
    models, man_seg, images = load_data()

    # =======================================================
    # PART 1 — Manual segmentation warm-up (man_seg)
    # =======================================================
    show_manual_segmentation(man_seg, images)

    # ---- Resample the manual seg to 14 points (warm-up only)
    print("\nResampling manual segmentation (man_seg) to 14 points...")
    resampled = resample_landmarks(man_seg, n_points=14)

    plt.figure()
    plt.imshow(images[:, :, 0], cmap="gray")
    plt.gca().invert_yaxis()
    draw_shape(resampled, ".-g")
    plt.title("Resampled 14-point Segmentation of man_seg")
    plt.axis("equal")
    plt.show()
    

    # =======================================================
    # PART 2 — GENERALIZED PROCRUSTES ALIGNMENT (Using models.mat)
    # =======================================================

    print("\nRunning Generalized Procrustes Alignment on 40 shapes...")

    # IMPORTANT: Copy models to avoid modifying the original data
    shapes_for_model = models.copy()   # <---- FIXED

    aligned_shapes, mean_shape = generalized_procrustes(shapes_for_model)

    print("Aligned shapes shape:", aligned_shapes.shape)
    print("Mean shape shape:", mean_shape.shape)

    plt.figure()
    draw_shape(mean_shape, ".-r")
    plt.title("Mean Shape After Procrustes Alignment")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.show()


    # =======================================================
    # PART 3 — PCA SHAPE MODEL
    # =======================================================
    print("\nBuilding PCA shape model...")

    mean_vec, eigvecs, eigvals = build_shape_model(aligned_shapes)

    print("First 5 eigenvalues:")
    print(eigvals[:5])

    # ---- Scree plot
    plot_eigenvalues(eigvals)

    # ---- Plot first 3 PCA modes
    show_modes(mean_vec, eigvecs, eigvals, num_modes=5, k=2)
    
    # ---------------------------------------------------
    # Decide how many modes to keep (e.g. 95% of energy)
    # ---------------------------------------------------
    t95, ratios = choose_num_modes(eigvals, target_energy=0.85)
    print(f"\nNumber of modes for 95% energy: t = {t95}")
   # for i in range(t95):
       # print(f"  Mode {i+1}: cumulative energy = {ratios[i]*100:.1f}%")

    # You can choose what you want to use in ASM:
    N_MODES_ASM = t95      # t95 or e.g. 3, 4, 6 if you want to experiment



    # =======================================================
    # PART 4 — Segmentation with the Shape Model (ASM)
    # =======================================================
    print("\nRunning ASM segmentation on 5 test images (21–25)...")

    # Convert mean_vec (2N,) back to complex mean shape for ASM
    mean_shape_cpx = vec_to_complex(mean_vec)

    # We'll segment images 21–25 (0-based indices 20..24)
    test_indices = [20, 21, 22, 23, 24]

    for idx in test_indices:
        img = images[:, :, idx]

        final_landmarks, init_mask, boundary_xy = asm_segment_single_image(
            img,
            mean_shape_cpx=mean_shape_cpx,
            mean_vec=mean_vec,
            eigvecs=eigvecs,
            eigvals=eigvals,
            n_modes=N_MODES_ASM,
            max_iters=25,
            conv_thresh=0.1,
            verbose=True,
        )

        # Visualize result
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap="gray")

        # Initial rough segmentation contour (yellow)
        plt.contour(init_mask, levels=[0.5], colors='y', linewidths=1)

        # Final ASM landmarks (red)
        draw_shape(final_landmarks, ".-r")

        plt.title(f"Shape model segmentation – image {idx+1}")
        plt.axis("equal")
        plt.gca().invert_yaxis()
        plt.show()

    print("------------PROGRAM ENDED------------")



if __name__ == "__main__":
    main()
