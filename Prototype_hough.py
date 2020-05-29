import cv2
import tools.data_manager as data_manager
import pickle
import tools.image_utils as img
from matplotlib import pyplot as plt
import numpy as np

""" first we need to decide if we want to recompute our data with different
parameters """

recompute_data = False

""" in this chapter we handle the preprocessing of our images, loading,
cropping and normalizing """
reload_images = False
if reload_images or recompute_data:
    # load, crop and normalize our images and store them
    # into a dictionary {patient_label:[array_pre, array_post]}
    print("load data from .png files")
    dict_data = data_manager.read_pictures()
    with open("dict_data.bin", "wb") as bin_file:
        pickle.dump(dict_data, bin_file)
else:
    print("load data from pickle")
    with open("dict_data.bin", "rb") as bin_file:
        dict_data = pickle.load(bin_file)

recompute_binaries = False
if recompute_binaries or recompute_data:
    print("generate binaries from dictionary")
    list_binaries = img.calculate_binaries(dict_data)
    with open("list_binaries.bin", "wb") as bin_file:
        pickle.dump(list_binaries, bin_file)
else:
    print("load binaries from pickle")
    with open("list_binaries.bin", "rb") as bin_file:
        list_binaries = pickle.load(bin_file)

# list_contours = [img.find_contours(image_binary) for image_binary in list_binaries]

shape_original = dict_data["03"][0].shape
accumulator_image = np.zeros(shape=shape_original)
list_edges = [img.edge_map(image_binary, 5) for image_binary in list_binaries]

img.plot_preprocessed_image(list_edges)

image = dict_data["03"][0]
plt.imshow(image)

a = 1
b = -20


theta = (np.linspace(0, np.pi * 2, 500))
x_2 = (a + b * theta) * np.cos(theta)
y_2 = (a + b * theta) * np.sin(theta)


center_x = 717
center_y = 373

x_2 += center_x
y_2 += center_y

plt.plot(x_2, y_2, "r")


for x, y in zip(x_2, y_2):
    print(x, y)
plt.show()


hough_array = np
vector_y, vector_x = np.where(image)  # find the indices of edge pixels
zip_edge_pixel_indices = zip(vector_y, vector_x)  # zip it for iterating

vector_theta = np.radians(np.linspace(0, 360, 16))

for y, x in zip_edge_pixel_indices:
    vector_r = [np.radians(y) + np.radians(x) * (theta ** 2) for theta in vector_theta]
    plt.plot(vector_theta, vector_r)
plt.show()



img.show(image)
