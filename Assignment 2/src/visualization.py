import numpy as np
import matplotlib.pyplot as plt

def draw_shape(points, style="-or"):
    """
    Draws a shape given complex-valued landmark points.
    PARAM points: 1D array of complex numbers
    PARAM style: matplotlib line/marker style
    """
    pts = np.asarray(points).squeeze()
    x = pts.real
    y = pts.imag

    plt.plot(x, y, style, linewidth=1.5)
    plt.plot([x[-1], x[0]], [y[-1], y[0]], style, linewidth=1.5)


def show_manual_segmentation(man_seg, images, pat_nbr=1):
    """Equivalent to MATLAB Part 1 visualization."""
    img = images[:, :, pat_nbr - 1]

    plt.figure()
    plt.imshow(img, cmap="gray", origin="upper")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    draw_shape(man_seg.squeeze(), ".-r")
    plt.title("Manual segmentation (man_seg)")
    plt.show()


def show_model_shapes(models, images, pat_nbr=1):
    """Equivalent to MATLAB Part 2 visualization."""
    img = images[:, :, pat_nbr - 1]

    plt.figure()
    plt.imshow(img, cmap="gray", origin="upper")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")

    # Right kidney
    right = models[:, pat_nbr - 1]
    draw_shape(right, ".-r")

    # Left kidney (mirrored)
    left = models[:, pat_nbr - 1 + 20]
    img_width = img.shape[1]
    mirrored_x = (img_width + 1) - left.real
    mirrored = mirrored_x + 1j * left.imag
    draw_shape(mirrored, ".-b")

    plt.title("Right kidney (red), Left kidney mirrored (blue)")
    plt.show()
