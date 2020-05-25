import numpy as np
import glob
from imageio import imread

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)

def read_pictures():
  
    # load the data in a dictionary where each key is the patients ID and the first
    # image is the pre_img and the second image is the post_img
    image_path = glob.glob("./DATA/ID*/ID*.png")

    data = {}  # this is our data dictionary
    patient_IDs = []  # this list is used for indexing during the crop

    for i, image in enumerate(image_path):
      patient_ID = image[9:11]
      # build list of patient ID's
      if patient_ID not in patient_IDs:
        patient_IDs.append(patient_ID)
    
      # read pictures into arrays
      post_img = np.array(imread("DATA/ID"+patient_ID+"/ID"+patient_ID+"post.png"))
      pre_img = np.array(imread("DATA/ID"+patient_ID+"/ID"+patient_ID+"pre.png"))
      
      # Some pictures are somehow 3 dimensional. We just take the first 2D picture.
      if pre_img.ndim == 3:
        pre_img = pre_img[:,:,0].astype("uint8")
      if post_img.ndim == 3:
        post_img = post_img[:,:,0].astype("uint8")
      
      data[patient_ID] = [pre_img, post_img]  # add images to dictionary
    
    
    # Crop images and normalize intensities
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
            
            #apply histogram equalization instead of the normalization above:
            #data[patient][i] = image_histogram_equalization(data[patient][i])
    
    return data