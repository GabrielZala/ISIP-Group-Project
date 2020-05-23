from matplotlib import pyplot as plt
from skimage import feature, color
import copy


def edge_map(image, sigma):
    # Returns the edge map of a given image.
    #
    # Inputs:
    #   img: image of shape (n, m, 3) or (n, m)
    #
    # Outputs:
    #   edges: the edge map of image

    img_grayscale = color.rgb2gray(image)
    img_edges = feature.canny(img_grayscale, sigma)

    return img_edges


def show(image_array):
    plt.imshow(image_array)
    plt.show()


def plot_image_list(list_images):
    # get how many images you want to plot horizontally
    nCols = len(list_images)
    nRows = 1

    # commence plotting
    for index_image, image in enumerate(list_images, 1):  # start at 1 because of shitty pyplot indexing
        plt.subplot(nRows, nCols, index_image)  # open subplot space to plot to
        plt.imshow(image)  # plot the image into subplot space

    plt.show()  # show plotted images


def test_sigmas(data, testing_range=range(10, 15, 1), pre_or_post=0, patient="03"):
    sample_image = data[patient][pre_or_post]

    list_image_edge_maps = []

    for sigma in testing_range:
        edge_map_current_sigma = edge_map(sample_image, sigma)
        list_image_edge_maps.append(edge_map_current_sigma)

    plot_image_list(list_image_edge_maps)


def data_to_edges(dict_data, sigma_pre, sigma_post):
    dict_data_edges = {}

    for patient in dict_data:
        # read images
        image_pre = dict_data[patient][0]
        image_post = dict_data[patient][1]

        # create their respective edge maps
        edge_map_pre = edge_map(image_pre, sigma_pre)
        edge_map_post = edge_map(image_post, sigma_post)

        dict_data_edges[patient] = [edge_map_pre, edge_map_post]  # add to edge dictionary

    return dict_data_edges


def create_material_masks(dict_data):

    # set thresholds for bone/fluid/gas
    t_bone = 150
    t_gas = 40

    for patient in dict_data:

        image_pre = dict_data[patient][0]  # [600:640, 540:600]
        image_post = dict_data[patient][1]

        image_pre_bone = copy.copy(image_pre)
        image_pre_fluid = copy.copy(image_pre)
        image_pre_gas = copy.copy(image_pre)

        for i_rows, row in enumerate(image_pre):
            for i_cols, value in enumerate(row):
                if value >= t_bone:
                    image_pre_bone[i_rows, i_cols] = 1
                    image_pre_fluid[i_rows, i_cols] = 0
                    image_pre_gas[i_rows, i_cols] = 0

                elif value <= t_gas:
                    image_pre_bone[i_rows, i_cols] = 0
                    image_pre_fluid[i_rows, i_cols] = 0
                    image_pre_gas[i_rows, i_cols] = 1

                else:
                    image_pre_bone[i_rows, i_cols] = 0
                    image_pre_fluid[i_rows, i_cols] = 1
                    image_pre_gas[i_rows, i_cols] = 0

        plot_image_list([image_pre_bone,image_pre_fluid,image_pre_gas])

    return None
