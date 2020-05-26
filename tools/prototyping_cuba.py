import tools.image_utils as img


def test_sigmas(data, testing_range=range(10, 15, 1), pre_or_post=0, patient="03"):  # WIP

    sample_image = data[patient][pre_or_post]

    list_image_edge_maps = []

    for sigma in testing_range:
        edge_map_current_sigma = img.edge_map(sample_image, sigma)
        list_image_edge_maps.append(edge_map_current_sigma)

    img.plot_image_list(list_image_edge_maps)

