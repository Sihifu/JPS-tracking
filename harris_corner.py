import numpy as np
import cv2 as cv
from background_subtraction_gmm import Backsub
from smoothen_mask import smooth_mask


def harris(image, mask):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    mask=smooth_mask(mask,open=False, close=False, erode=True)
    H=cv.cornerHarris(gray)
    H[mask==0]=0
    return H

if __name__=="__main__":
    image=cv.imread(cv.samples.findFile("Image_Data/test_frame26.jpg"), cv.IMREAD_COLOR)
    backsub=Backsub("Video_Data/test_video.avi")
    backsub.begin_train(max_frames=500)
    mask=backsub.getMask(image)
    foreground=np.copy(image)
    foreground[mask==0]=0
    H=harris(image,mask)
    foreground[H>0.005*H.max()]=[0,0,255]
    image[H>0.005*H.max()]=[0,0,255]
    cv.imshow('features',image)
    cv.waitKey(0)
    cv.destroyAllWindows()

