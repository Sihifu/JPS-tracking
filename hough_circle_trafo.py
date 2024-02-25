import cv2 as cv
import numpy as np


def hough_circle(image, draw=False, round=False, dp=1, min_dist=40, canny_upper=80, canny_lower=10, minradius=5, maxradius=25):
    """
    Detecs circles in the image.
    Returns the detected n circles in a numpy array of shape (n,3)
    The values along the 2nd dimension are the center of the detected circle (x and y coordinate).
    The last value is the radius of the circle.

    Parameters
    ----------
    image : np.array
        The image array as bgr tensor
    draw : boolean
        if true draws and shows the detected circle
    round : boolean
        Rounds the output to next uint16
    dp: inverse ratio of accumulator matrix
    mindist: min dist between detected circles
    canny_upper: upper canny threshhold (strong edge)
    canny_lower: lower canny threshhold (weak edge)
    minradius: minimum radius of circles
    maxradius: maximum radius of circles
    return: circles numpy.array of shape (n,3)
        n is number of circles detected, row represents x coordinate y coordinate and radius of circles 
    """

    # transform image to grayscale

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # smooth with median filter, gaussian filter also fine window size 3
    # gray = cv.medianBlur(gray, 5)
    gray = cv.GaussianBlur(gray, ksize=(3,3),sigmaX=0)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=canny_upper, param2=canny_lower, minRadius=minradius, maxRadius=maxradius)
    if circles is None:
        return circles
    if round:
            circles = np.uint16(np.around(circles))
    if draw:
        if circles is not None:
            circles = np.uint16(np.around(circles))[0,:]
        for i in circles:
            center = (i[0], i[1])
            # circle outline
            radius = i[2]
            # circle center
            cv.circle(image, center, 1, (0, 100, 100), 3)
            cv.circle(image, center, radius, (255, 0, 255), 3)

        cv.imshow("detected circles", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    circles=circles.reshape((-1,3))
    return circles

if __name__ == "__main__":
    # Loads an image
    #im = cv.imread(cv.samples.findFile("Image_Data/test_images/test_frame26.jpg"), cv.IMREAD_COLOR)
    #im=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t011.jpg"), cv.IMREAD_COLOR)
    #im=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t279.jpg"), cv.IMREAD_COLOR)
    im=cv.imread("Image_Data/Channel_19_10_2023/Take_04/Project_Channel_take_04_t1402.jpg", cv.IMREAD_COLOR)
    #im=cv.imread("/Users/Hoang_1/Desktop/Master_Arbeit/software/janus_particle_tracking/Image_Data/test_images/test_frame452.jpg", cv.IMREAD_COLOR)
    circles=hough_circle(im,draw=True,houghparameter=[1,40,80,10,13,19])
    if circles!=None:
        print(circles.shape)
    else:
        print("no circles")