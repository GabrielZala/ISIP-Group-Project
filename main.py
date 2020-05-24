import tools.data_manager_sandro as data_manager
import pickle
import tools.image_utils_cuba as img
import tools.prototyping_cuba as prototyper
import tools.methods_circle_detection_cudi as methods


""" first we need to decide if we want to recompute our data with different
parameters """

recompute_data = False

""" in this chapter we handle the preprocessing of our images, loading,
cropping and normalizing """
dict_data = None
if recompute_data:
    # load, crop and normalize our images and store them
    # into a dictionary {patient_label:[array_pre, array_post]}
    dict_data = data_manager.read_pictures()
    with open("dict_data.bin", "wb") as bin_file:
      pickle.dump(dict_data, bin_file)
else:
    with open("dict_data.bin", "rb") as bin_file:
      pickle.load(dict_data, bin_file)

""" here i tried to estimate an appropriate sigma for our edge detection """

# estimate "sigma" for canny edge map method - smoothing factor, 0 for pre, 1 for post
# prototyper.test_sigmas(dict_data, range(15, 18, 1), 0)  # range(15, 18, 1) shows potentially good results for pre_images
# prototyper.test_sigmas(dict_data, range(9, 16, 3), 1)  # not so sure if we can use it for post images tho


""" in this chapter i transform the dictionary """

# choose sigma for both pre and post images
sigma_pre = 40
sigma_post = 40

recompute_image_edges = True
if recompute_image_edges or recompute_data:
    # create edge maps of images
    dict_data_edges = img.data_to_edges(dict_data, sigma_pre, sigma_post)
    pickle.save(dict_data_edges, "dict_data_edges")

else:
    dict_data_edges = pickle.load("dict_data_edges")

# now merge the image and edge dictionaries WIP
platzhalter = data_manager.merge_dicts(dict_data, dict_data_edges)

""" new concept for edges - split images into bone air fluid channels """

img.create_material_masks(dict_data)

""" here we attempt to find circles in our image, once with image and once with it's edge map """
image = dict_data["05"][0]
edges = dict_data_edges["05"][0]

#circles_image = methods.circles_find(image)
circles_edges = methods.circles_find(edges)

#spiral_center_image = methods.get_center(circles_image)
spiral_center_edges = methods.get_center(circles_edges)

#methods.circles_show(image, spiral_center_image)
methods.circles_show(image, spiral_center_edges)
