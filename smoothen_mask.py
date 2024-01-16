from k_means_clustering import k_means_clustering
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from cake import cake

def smooth_mask(mask, kernel_size=4, open=True, close=True, erode=False, dilate=False):
    """
    Smoothens a background foreground mask (1 is foreground) with closing

    Parameters
    ----------
    mask : np.array (n,m) with values 0 or 1
    
    return: smoothened mask
    """
    mask=mask.astype(np.uint8)
    kernel=cake(kernel_size).astype(np.uint8)
    if open==True:
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    if close==True:
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    if erode==True:
        mask = cv.erode(mask, kernel)
    if dilate==True:
        mask = cv.dilate(mask, kernel)

    return mask
    
if __name__=="__main__":
    center, mask = k_means_clustering("Image_Data/test_frame1939.jpg", K=3)
    mask=mask==1
    mask=mask.astype(np.uint8)*255
    mask=smooth_mask(mask)
    src=cv.imread("Image_Data/test_frame1939.jpg")
    fig = plt.figure()
    # setting values to rows and column variables
    rows = 1
    columns = 2
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    
    # showing image
    plt.imshow(src)
    plt.axis('off')
    plt.title("source")
    fig.add_subplot(rows, columns, 2)
    
    # showing image
    plt.imshow(mask)
    plt.axis('off')
    plt.title("mask")
    plt.show()