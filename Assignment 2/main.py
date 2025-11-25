import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Your modules
from src.visualization import (
    show_manual_segmentation,
    draw_shape
)

from src.resampling import resample_landmarks
from src.procrustes import generalized_procrustes
from src.shape_model import build_shape_model, plot_eigenvalues, show_modes, complex_to_vec


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
    plt.title("Resampled 14-point Segmentation (Warm-Up)")
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

    print("------------PROGRAM ENDED------------")



if __name__ == "__main__":
    main()
