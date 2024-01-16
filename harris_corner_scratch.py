import numpy as np
import cv2 as cv
from scipy.signal import convolve2d as convolve2d
from background_subtraction_gmm import Backsub
from smoothen_mask import smooth_mask
from cake import cake
import matplotlib.pyplot as plt


"""
Harris Corneter detector, returns the coordniates of the features 

Returns
----------
features : np.array (2,n) with n features and first dimension is the y coordinate
"""
def harris(image, mask):
    segment_length = 15
    k = 0.05
    tau = 10e6
    min_dist = 10
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    Kx = np.array([[1, 0, -1], 
               [2, 0, -2], 
               [1, 0, -1]])

    Ky = np.array([[1, 2, 1], 
               [ 0,  0,  0], 
               [ -1,  -2,  -1]])
    IX= convolve2d(gray,Kx,mode='same')
    IY= convolve2d(gray,Ky,mode='same')
    G11=np.power(IX,2).astype(np.float32)
    G22=np.power(IY,2).astype(np.float32)
    G12=np.multiply(IX,IY).astype(np.float32)
    G11=cv.GaussianBlur(G11,(segment_length, segment_length), segment_length/2)
    G12=cv.GaussianBlur(G12,(segment_length, segment_length), segment_length/2)
    G22=cv.GaussianBlur(G22,(segment_length, segment_length), segment_length/2)
    H=np.multiply(G11,G22)-np.power(G12,2) - k * np.power(G11+G22,2)

    mask=smooth_mask(mask, open=False, close=False, erode=True)
    corners=np.ones(H.shape)
    corners[mask==0]=0
    corners[H<tau]=0
    corners=np.pad(corners,pad_width=min_dist)
    H=np.pad(H,pad_width=min_dist)
    sorted_index = np.unravel_index(np.argsort(H,axis=None), H.shape)
    number_of_features=np.sum(H!=0)
    y=sorted_index[0]
    y=y[-1:-1-number_of_features:-1]
    x=sorted_index[1]
    x=x[-1:-1-number_of_features:-1]
    sorted_index=np.vstack((y,x))
    features=np.zeros_like(sorted_index)
    

    M=cake(min_dist)==0
    count=0
    for coordinates in sorted_index.transpose():
            y=coordinates[0]
            x=coordinates[1]
            if corners[y,x]!=0:
                corners[y-min_dist:y+min_dist+1,x-min_dist:x+min_dist+1]=np.multiply(corners[y-min_dist:y+min_dist+1,x-min_dist:x+min_dist+1],M)
                features[:,count]=np.vstack((y,x)).reshape(-1)
                count+=1

    features=features[:,:count]-min_dist
    return features

if __name__=="__main__":
    image=cv.imread(cv.samples.findFile("Image_Data/test_frame26.jpg"), cv.IMREAD_COLOR)
    backsub=Backsub("Video_Data/test_video.avi")
    backsub.begin_train(max_frames=500)
    mask=backsub.getMask(image)
    foreground=np.copy(image)
    foreground[mask==0]=0
    features=harris(foreground,mask)

    # Radius of circle
    radius = 3
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2
    for i in range(features.shape[1]):
        center=(features[1,i],features[0,i])
        image = cv.circle(image, center, radius, color, thickness)
    cv.imshow("foreground",image)
    cv.waitKey(0)
    cv.destroyAllWindows()