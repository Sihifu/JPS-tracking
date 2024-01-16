import cv2 as cv
import numpy as np

def k_means_clustering(filename, K=2, draw=False):
    """
    Perfroms K-means clustering on image in grayscale
    Returns centroid and labels of the image

    Parameters
    ----------
    filename : str
        The path to the image
    K : int
        Clusters (standard value 2 into background and rest)
    draw : boolean
        if true draws the segmentation result in grayscale
    
    
    returns:
        center: centroid np.array with shape (K,1)
        mask: np.array mask with labels (K integer values) of grayscale image shape  
    """
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray_vector = np.float32(gray.reshape((-1)))
  
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 ) for termination, in this case max iter and eps cirteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts=10
    ret,label,center=cv.kmeans(gray_vector, K, None, criteria, attempts, cv.KMEANS_RANDOM_CENTERS)
    mask=label.reshape(gray.shape)
    if draw:
        # Convert back into uint8, and draw original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((gray.shape))
        cv.imshow('res2',res2)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imshow('mask',np.uint8(255/(K-1)*mask))
        cv.waitKey(0)
        cv.destroyAllWindows()
    return center, mask

if __name__=="__main__":
    #center, mask=k_means_clustering("Image_Data/test_frame100.jpg", K=3, draw=True)
    src = cv.imread(cv.samples.findFile("Image_Data/test_images/test_frame1939.jpg"), cv.IMREAD_COLOR)
    cv.imshow('mask',src)
    cv.waitKey(0)
    cv.destroyAllWindows()
    center, mask=k_means_clustering("Image_Data/test_images/test_frame1939.jpg", K=3, draw=True)
    cv.imshow('mask',mask)
    cv.waitKey(0)
    cv.destroyAllWindows()