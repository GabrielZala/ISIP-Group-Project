import numpy as np
import glob
from imageio import imread
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

# load the data in a dictionary where each key is the patients ID and the first
# image is the pre_img and the second image is the post_img
image_path = glob.glob("./DATA/ID*/ID*.png")
data = {}
patient_IDs = []
for i, image in enumerate(image_path):
  patient_ID = image[9:11]
  if patient_ID not in patient_IDs:
    patient_IDs.append(patient_ID)
  post_img = np.array(imread("DATA/ID"+patient_ID+"/ID"+patient_ID+"post.png"))
  pre_img = np.array(imread("DATA/ID"+patient_ID+"/ID"+patient_ID+"pre.png"))
  data[patient_ID] = [pre_img, post_img]


# I realized that the watermark always goes exactly down to the 23th pixel
# with the following code:
if 0:
  for patient in patient_IDs:  
    plt.imshow(data[patient][0][0:24, 0:50])
    plt.show()
    plt.imshow(data[patient][0][0:24, -50:-1])
    plt.show()
    plt.imshow(data[patient][1][0:24, 0:50])
    plt.show()
    plt.imshow(data[patient][1][0:24, -50:-1])
    plt.show()
    

for patient in patient_IDs:
  for i in range(len(data[patient])): # i = [0, 1] -> pre_img, post_img
    # Crop away the first 23 pixelrows of every picture
    data[patient][i] = data[patient][i][23:, :]
    
    ### normalize each individual pic to again hold values from 0 to 255
    # get the min and max values and the resulting range
    min_val = np.min(data[patient][i])
    max_val = np.max(data[patient][i])
    val_range = max_val - min_val
    # calculate new values
    data[patient][i] = ((data[patient][i] - min_val) / val_range) * 255
  
  # Old style (pre and post individually):
  # # Crop away the first 23 pixelrows of every picture
  # data[patient][0] = data[patient][0][23:, :]
  # data[patient][1] = data[patient][1][23:, :]
  # ### normalize each individual pic to hold values from 0 to 255
  # # get the min and max values
  # min0 = np.min(data[patient][0])
  # max0 = np.max(data[patient][0])
  # min1 = np.min(data[patient][1])
  # max1 = np.max(data[patient][1])
  # # store the range, calculate new values and assign them
  # range0 = max0 - min0
  # range1 = max1 - min1
  # data[patient][0] = ((data[patient][0] - min0) / range0) * 255
  # data[patient][1] = ((data[patient][1] - min1) / range1) * 255
  # # Verify that all pictures have values spaning from 0 to 255
  # if 1:
  #   print("\n  patient:", patient)
  #   print("old pre:  " + str(min0) + " to " + str(max0))
  #   print("old post: " + str(min1) + " to " + str(max1))  
  #   min0 = np.min(data[patient][0])
  #   max0 = np.max(data[patient][0])
  #   min1 = np.min(data[patient][1])
  #   max1 = np.max(data[patient][1])
  #   print("new pre:  " + str(int(min0)) + " to " + str(int(max0)))
  #   print("new post: " + str(int(min1)) + " to " + str(int(max1)))

from utils import binarize_image
#test_img = binarize_image(data["03"][0], 130, 180)


### circle detection ###
import cv2
import numpy as np
from utils import hough_circle
img = data["04"][0]
circles = hough_circle(img)
cimg = cv2.cvtColor(img.astype("uint8"),cv2.COLOR_GRAY2BGR)   
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()