import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

col1 = "Collection 1"
HE_1 = os.path.join(col1, "HE")
AMACR_1 = os.path.join(col1, "p63AMACR")

import cv2
import numpy as np

# Load 
img1 = cv2.imread("Collection 1/HE/1.1.bmp")
img2 = cv2.imread("Collection 1/p63AMACR/1.1.bmp")

# grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


def makePairHalfSize(image1, image2):
    # Get dimensions
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Calculate new dimensions
    new_width1, new_height1 = width1 // 2, height1 // 2
    new_width2, new_height2 = width2 // 2, height2 // 2

    # Resize images
    resized_image1 = cv2.resize(image1, (new_width1, new_height1))
    resized_image2 = cv2.resize(image2, (new_width2, new_height2))

    return resized_image1, resized_image2
# half size
gray1, gray2 = makePairHalfSize(gray1, gray2)

def plotImagepair(image1, image2, title1="Image 1", title2="Image 2"):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title(title2)
    plt.axis("off")

    plt.show()

# Display the images
plotImagepair(gray1, gray2, title1="Image 1 - HE", title2="Image 2 - AMACR")

# testing
#hello0
