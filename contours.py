import cv2 as cv
import numpy as np
from background_subtraction_gmm import Backsub
from smoothen_mask import smooth_mask
from matplotlib import pyplot as plt

def draw_rectangular(image,contours):
    for i in range(len(contours)):
        x,y,w,h = cv.boundingRect(contours[i])
        cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    return image


def contours(mask):
    """
    Get contours of individual object from binary mask
    Returns contours and hierachy

    Parameters
    ----------
    filename : np.array (n,m) with values 0 1 
        The path to the image
    returns:
        contours: tuple with length of number of objects, each elemt of tuple contain array with pixels of contour 
    """
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours


def calculate_centroids(contours):
    M=np.zeros((len(contours),2))
    for i in range(len(contours)):
        m=cv.moments(contours[i])
        cx = int(m['m10']/m['m00'])
        cy = int(m['m01']/m['m00'])
        M[i,:]=np.array([cy,cx])
    return M 


def get_segmentation_mask(mask,contours):
    segmentation_mask=np.zeros(mask.shape)
    for i in range(len(contours)):
        x,y,w,h = cv.boundingRect(contours[i])
        segmentation_mask[y:y+h,x:x+w]=i+1
    segmentation_mask=np.multiply(segmentation_mask,mask)
    return segmentation_mask

if __name__=="__main__":
    im=cv.imread(cv.samples.findFile("Image_Data/test_images/test_frame26.jpg"), cv.IMREAD_COLOR)
    backsub=Backsub("Video_Data/test_video.avi")
    backsub.begin_train(max_frames=500)
    mask=backsub.getMask(im)
    cont=contours(mask)
    cv.drawContours(im, cont, 0, (255,0,0), 3)
    cv.drawContours(im, cont, 1, (0,255,0), 3)
    cv.drawContours(im, cont, 2, (0,0,255), 3)
    im=draw_rectangular(im,cont)
    M=calculate_centroids(cont)
    for i in M:
        x=int(i[1])
        y=int(i[0])
        im = cv.circle(im, (x,y), radius=2, color=(100, 255, 100), thickness=-1)
    cv.imshow("im",im)
    cv.waitKey(0)
    cv.destroyAllWindows()
    segmentation_mask=get_segmentation_mask(mask,cont)
    plt.figure()
    plt.imshow(segmentation_mask)
    plt.show()