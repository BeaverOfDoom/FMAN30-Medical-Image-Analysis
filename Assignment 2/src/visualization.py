import numpy as np
import matplotlib.pyplot as plt

def draw_shape(points, style="-or"):
    
    pts = np.asarray(points).squeeze()
    x = pts.real
    y = pts.imag

    plt.plot(x, y, style, linewidth=1.5)
    plt.plot([x[-1], x[0]], [y[-1], y[0]], style, linewidth=1.5)


def show_manual_segmentation(man_seg, images):
    
    img = images[:, :, 0]

    plt.figure()
    plt.imshow(img, cmap="gray", origin="upper")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    draw_shape(man_seg.squeeze(), ".-r")
    plt.title("Manual segmentation (man_seg)")
    plt.show()


