import numpy as np
import glob
from imageio import imread
from matplotlib import pyplot as plt
import cv2

def read_pictures():
  
    plt.rcParams['image.cmap'] = 'gray'

    # load the data in a dictionary where each key is the patients ID and the first
    # image is the pre_img and the second image is the post_img
    image_path = glob.glob("./DATA/ID*/ID*.png")

    data = {}  # this is our data dictionary
    patient_IDs = []  # this list is used for indexing during the crop

    for i, image in enumerate(image_path):

        patient_ID = image[9:11]  # gets the patient id string

        if patient_ID not in patient_IDs:
            patient_IDs.append(patient_ID)  # build list of patient ID's

        # read pictures into arrays
        post_img = np.array(cv2.imread("DATA/ID" + patient_ID + "/ID" + patient_ID + "post.png"))
        pre_img = np.array(cv2.imread("DATA/ID" + patient_ID + "/ID" + patient_ID + "pre.png"))

        """ this deals with inconsistencies in the dataset """
        # now this is stupid, the png's have different data structures which lead some of them to use arrays instead
        if type(pre_img[0, 0]) == type(pre_img):  # of uint8's to save pixel data, this finds them and corrects.

            # create array that we use to overwrite our image later
            nRows = np.shape(pre_img)[0]
            nCols = np.shape(pre_img)[1]
            array_temp = np.zeros(shape=(nRows, nCols))

            # pick only the 1st value from the list [value, value, value, 255]
            for i_rows, row in enumerate(pre_img):
                for i_cols, list_values in enumerate(row):
                    array_temp[i_rows, i_cols] = list_values[0]  # just pick the 1st value of the list

            pre_img = array_temp.astype("uint8") # so it retains the same datatype as other arrays in the dict

        # now this is stupid, the png's have different data structures which lead some of them to use arrays instead
        if type(post_img[0, 0]) == type(post_img):  # of uint8's to save pixel data, this finds them and corrects.

            # create array that we use to overwrite our image later
            nRows = np.shape(post_img)[0]
            nCols = np.shape(post_img)[1]
            array_temp = np.zeros(shape=(nRows, nCols))

            # pick only the 1st value from the list [value, value, value, 255]
            for i_rows, row in enumerate(post_img):
                for i_cols, list_values in enumerate(row):
                    array_temp[i_rows, i_cols] = list_values[0]  # just pick the 1st value of the list

            post_img = array_temp.astype("uint8")  # so it retains the same datatype as other arrays in the dict

        data[patient_ID] = [pre_img, post_img]  # add images to dictionary

    # Let the cropping begin
    for patient in patient_IDs:
        for i in range(len(data[patient])):  # i = [0, 1] -> pre_img, post_img

            # Crop away the first 23 pixelrows of every picture
            data[patient][i] = data[patient][i][23:, :]

            # normalize each individual pic to again hold values from 0 to 255
            # get the min and max values and the resulting range
            min_val = np.min(data[patient][i])
            max_val = np.max(data[patient][i])
            val_range = max_val - min_val

            # calculate new values
            data[patient][i] = ((data[patient][i] - min_val) / val_range) * 255
    return data
