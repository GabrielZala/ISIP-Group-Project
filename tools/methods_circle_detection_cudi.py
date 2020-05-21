import cv2  # conda install -c menpo opencv
import numpy as np
import tools.image_utils_cuba as img


# binarize image
def binarize_image(image, lower_threshold, upper_threshold):
    mask = (image > lower_threshold) & (image < upper_threshold)  # sets all values to 0 that are not within
    image = image * mask  # within thresholds
    image[image > 0] = 255  # set all values to 255 that are above 0 (black in the image)
    return image


def hough_circle(image):
    """
    Hough Circle Transform:
        first number in function: dp=1, resolution factor between image and detection
        second number : minDist = minimum number of pixels between circle centers
        param1: number used for canny edge detection
        param2: threshold for circle detection (small values=more circles will be detected)
        min/maxRadius: only detects circles that are between those values
    """

    # run circle detection algo
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=60, minRadius=0, maxRadius=0)

    """print(type(circles))  # just some data on what's found
    print(circles)
    print(np.shape(circles))"""

    circles = np.uint16(np.around(circles))  # what the fuck is this

    return circles

    # just to visualize the circles on the original image
    # cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # for i in circles[0,:]:
    #     # draw the outer circle
    #     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    #     # draw the center of the circle
    #     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    # cv2.imshow('detected circles',cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def circles_find(image):

    if image.dtype == bool:  # edge maps are boolean and need to be converted first for the cv2 module
        image = image.astype("uint8")  # convert boolean to array with range [0, 1]

        # this is not enough tho, we need range [1, 255] so we loop over the whole thing and change the values
        nRows = np.shape(image)[0]
        nCols = np.shape(image)[1]
        for i in range(nRows):
            for j in range(nCols):
                if image[i, j] == 1:
                    image[i, j] = 255
                else:
                    image[i, j] = 1

    else:
        image = image.astype("uint8")  # also, not all numeric arrays are supported, so convert just in case
        image = cv2.medianBlur(image, 5)  # add some blur to reduce noise

    circles = hough_circle(image)

    return circles


def circles_show(image, circles):

    cimg = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_GRAY2BGR)

    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
