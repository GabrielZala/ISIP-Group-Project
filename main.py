import tools.data_manager_sandro as data_manager
import pickle
import tools.image_utils_cuba as img
import tools.prototyping_cuba as prototyper
import tools.methods_circle_detection_cudi as methods
from matplotlib import pyplot as plt
import cv2
import numpy as np

""" first we need to decide if we want to recompute our data with different
parameters """

recompute_data = True

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
      dict_data = pickle.load(bin_file)

""" here i tried to estimate an appropriate sigma for our edge detection """

# estimate "sigma" for canny edge map method - smoothing factor, 0 for pre, 1 for post
# prototyper.test_sigmas(dict_data, range(15, 18, 1), 0)  # range(15, 18, 1) shows potentially good results for pre_images
# prototyper.test_sigmas(dict_data, range(9, 16, 3), 1)  # not so sure if we can use it for post images tho


""" in this chapter i transform the dictionary """

# choose sigma for both pre and post images
sigma_pre = 40
sigma_post = 40

dict_data_edges = None
recompute_image_edges = False
if recompute_image_edges or recompute_data:
  # create edge maps of images
  dict_data_edges = img.data_to_edges(dict_data, sigma_pre, sigma_post)
  with open("dict_data_edges.bin", "wb") as bin_file:
    pickle.dump(dict_data_edges, bin_file)

else:
  with open("dict_data_edges.bin", "rb") as bin_file:
      dict_data_edges = pickle.load(bin_file)

# now merge the image and edge dictionaries WIP
#platzhalter = data_manager.merge_dicts(dict_data, dict_data_edges)

""" new concept for edges - split images into bone air fluid channels """

#img.create_material_masks(dict_data)

""" here we attempt to find circles in our image, once with image and once with it's edge map """
Hough_circle_detection = False

if Hough_circle_detection:
    #dict_data_cropped = methods.crop_images(dict_data, y0=150, y1=550, x0=600,x1=800)
    dict_of_centres = {}
    for i in dict_data:
        print(i)
        image = dict_data[i][1]
        try:    
            circles_image = methods.circles_find(image)
            dict_of_centres[i] = circles_image
            print(i,"in try", "number of circles found:",len(circles_image[0]))
        except:
            print(i,"returned None, ergo no circles found")
            pass
        

    for i in dict_of_centres: #use arrowkeys to go through the images
        methods.circles_show(dict_data[i][1],dict_of_centres[i])


findingCentre = True
""" this approach trys to find the electrodes using their intensities and contours.
maybe we could implement how far the points should be a part maximal and minimal to 
get rid of the wrong center points. some pictures would benefit from an image
erosion such as img 5. but i dont know how to choose these images beforehand"""

if findingCentre:
    result_dict ={}
    from scipy import ndimage as ndi
    for image in dict_data:
        data = dict_data[image][1]
        data = data.astype("uint8")
        img = cv2.medianBlur(data, 29)
        # find local max
        image_max = ndi.maximum_filter(img, size=5) #size gives the shape that is taken from the input array, at every element position, to define the input to the filter function.
        
        from tools.data_manager_sandro import image_histogram_equalization 
        histo = image_histogram_equalization(image_max)
        
        binary = methods.binarize_image(histo, lower_threshold=245, upper_threshold=255)
        #now find the contours to calculate their centres
        binary = cv2.convertScaleAbs(binary) #need to convert to special format...
        im2, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #calculating the centres of each contour
        lst_of_centres = []
        
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"]/ M["m00"])
                cY = int(M["m01"]/ M["m00"])
            else:
                cX, cY = 0, 0
            coords = [cX,cY]
            lst_of_centres.append(coords)
        print("number of centres found", len(lst_of_centres))
        for i in lst_of_centres:
            x = i[0]
            y= i[1]
            result=cv2.circle(data, (x, y), 5, (0, 0, 255), -1)
        result_dict[image]=result,lst_of_centres
    

for i in result_dict:
    plt.imshow(result_dict[i][0])
    plt.show()


"""
coni's Sandbox

from skimage.feature import peak_local_max

coordinates = peak_local_max(binary_max, min_distance=40)
kernel = np.ones((5,5), np.uint8)
img_erosion = cv2.erode(binary_max, kernel, iterations=4)
plt.imshow(img_erosion)






plt.imshow(histo)
plt.imshow(image_max)
plt.imshow(data)
plt.imshow(binary_max)


img = dict_data["03"][1]
img.astype("uint8")
img=cv2.GaussianBlur(img,(7,7),3)
ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
watershed = cv2.watershed(binary,markers)
plt.imshow(binary)
"""