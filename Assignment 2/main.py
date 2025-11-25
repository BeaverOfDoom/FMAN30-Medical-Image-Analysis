import os
from scipy.io import loadmat

# Import visualization functions
from src.visualization import show_manual_segmentation, show_model_shapes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_data():
    models = loadmat(os.path.join(DATA_DIR, "models.mat"))["models"]
    man_seg = loadmat(os.path.join(DATA_DIR, "man_seg.mat"))["man_seg"]
    dmsa_images = loadmat(os.path.join(DATA_DIR, "dmsa_images.mat"))["dmsa_images"]

    print("Loaded models:", models.shape)
    print("Loaded man_seg:", man_seg.shape)
    print("Loaded images:", dmsa_images.shape)

    return models, man_seg, dmsa_images


def main():
    print("------------STARTING PROGRAM------------")
    models, man_seg, images = load_data()

    # PART 1
    show_manual_segmentation(man_seg, images)

    # PART 2
    #show_model_shapes(models, images)


if __name__ == "__main__":
    main()
    print("------------PROGRAM ENDED------------")