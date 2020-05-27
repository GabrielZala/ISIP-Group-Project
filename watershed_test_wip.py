import sys
import cv2
import numpy
from scipy.ndimage import label
from matplotlib import pyplot as plt
import cv2
import tools.data_manager as data_manager
import pickle
import tools.image_utils as imut
import tools.prototyping_cuba as prototyper
import tools.methods_circle_detection as methods
from matplotlib import pyplot as plt
import tools.segmentation_kmeans as segmentator
import numpy as np

def info(image):
    print(type(image))
    print(image.dtype)
    print(image.shape)
    show(image)


def show(image):
    plt.imshow(image)
    plt.show()


def segment_on_dt(image_original, image_binary):
    # don't really know what this is for
    border = cv2.dilate(image_binary, None, iterations=2)
    border = border - cv2.erode(border, None)
    # info(border)

    # # create distance transform data
    # image_distance_transform = cv2.distanceTransform(image_binary, 2, 3)
    # info(image_distance_transform)

    # image_distance_transform = ((image_distance_transform - image_distance_transform.min()) / (image_distance_transform.max() - image_distance_transform.min()) * 255).astype(numpy.uint8)
    # info(image_distance_transform)

    # image_binary = cv2.threshold(image_distance_transform, 180, 255, cv2.THRESH_BINARY)[1]
    # info(image_binary)

    # making markers i think
    labels, ncc = label(image_binary)
    labels = labels * (255 / (ncc + 1))
    info(labels)

    # Completing the markers now.
    labels[border == 255] = 255
    labels = labels.astype(numpy.int32)  # convert to valid datatype for watershed
    info(labels)

    # run actual watershed algo
    cv2.watershed(image_original, labels)

    labels[labels == -1] = 0
    labels = labels.astype(numpy.uint8)
    return 255 - labels


image = cv2.imread("./data/ID05/ID05post.png")

# Pre-processing.
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ## Those are our preprocessing steps to get a good binary image ## #
""" evtl muama das auno ir funktion oba implementiara """
image_blurred = cv2.medianBlur(img_gray, 29)

image_segmented = segmentator.run_kmean_on_single_image(image_blurred, k=10, precision=10000, max_iterations=1000)

lower_threshold = imut.intelligent_get_threshold(image_segmented, fraction_of_image_threshold=0.08)

img_bin = cv2.threshold(img_gray, lower_threshold, 255, cv2.THRESH_OTSU)[1]
show(img_bin)

# #####################################################################

img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, numpy.ones((3, 3), dtype=int))

info(image)
info(img_bin)
result = segment_on_dt(image, img_bin)

result[result != 255] = 0
result = cv2.dilate(result, None)
image[result == 255] = (0, 0, 255)
info(image)