#binarize image
def binarize_image(image,lower_threshold,upper_threshold):
    mask = (image>lower_threshold) & (image < upper_threshold) #sets all values to 0 that are not within
    image = image * mask                                       #within thresholds
    image[image>0] = 255                                       #set all values to 255 that are above 0 (black in the image)    
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
    import cv2
    import numpy as np
    img = image.astype("uint8")
    img = cv2.medianBlur(img,5)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,10,
                                param1=50,param2=60,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    
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