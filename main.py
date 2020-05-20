import tools.data_manager_sandro as data_manager
import tools.data_pickler_cuba as pickle
import tools.image_utils_cuba as img

""" first we need to decide if we want to recompute our data with different parameters """

recompute_data = False

""" in this chapter we handle the preprocessing of our images, loading, cropping and normalizing """

if recompute_data:
    # load and crop and normalize our images into a dictionary {patient_label:[array_pre, array_post]}
    dict_data = data_manager.preprocess_data()
    pickle.save(dict_data, "dict_data")

else:
    dict_data = pickle.load("dict_data")

""" here i tried to estimate an appropriate sigma for our edge detection """

# estimate "sigma" for canny edge map method - smoothing factor, 0 for pre, 1 for post
img.test_sigmas(dict_data, range(15, 18, 1), 0)  # range(15, 18, 1) shows potentially good results for pre_images
img.test_sigmas(dict_data, range(9, 16, 3), 1)  # not so sure if we can use it for post images tho

""" in this chapter i transform the dictionary """

# choose sigma for both pre and post images
sigma_pre = 18
sigma_post = 12

if recompute_data:
    # create edge maps of images
    dict_data_edges = img.data_to_edges(dict_data, sigma_pre, sigma_post)
    pickle.save(dict_data_edges, "dict_data_edges")

else:
    dict_data_edges = pickle.load("dict_data_edges")

img.plot_image_list(dict_data_edges["55"])
