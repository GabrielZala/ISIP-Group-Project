from matplotlib import pyplot as plt
import numpy as np
import cv2


def run_kmean_on_single_image(image_array, k, precision=10, max_iterations=0.1):

    image_array = np.uint8(image_array)

    # blur image beforehand
    image_array = cv2.medianBlur(image_array, 11)

    # make the image flat
    image_flattened = image_array.reshape((-1, 3))

    # convert to np.float32
    image_flat_converted = np.float32(image_flattened)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, precision, max_iterations)

    # run k-means clustering
    ret, label, center = cv2.kmeans(image_flat_converted, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)

    res = center[label.flatten()]  # label is actually already flat, but do it anyways for consistency
    res2 = res.reshape(image_array.shape)

    return res2


def segment_img_data(dict_data, k_pre=3, k_post=3):

    Ks = [k_pre, k_post]

    dict_data_segmented = {}

    for patient in dict_data:

        segmented_images_current_patient = []

        for i in range(2):

            # get pre image & K-parameter
            image = dict_data[patient][i]
            K = Ks[i]

            segmented_image = run_kmean_on_single_image(image, K)  # generate segmented image

            segmented_images_current_patient.append(segmented_image)   # append segmented image to list

        dict_data_segmented[patient] = segmented_images_current_patient  # add patient images to data dictionary

    return dict_data_segmented
