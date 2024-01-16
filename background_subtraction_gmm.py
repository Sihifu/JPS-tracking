from __future__ import print_function
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from smoothen_mask import smooth_mask

"""
    Estimates the background from a video with gaussian mixture model
"""
class Backsub:
    def __init__(self, video_path, history=500, detectShadows=False):
        """
        Intialize backsub object and properties

        Parameters
        ----------
        video_path: str
            path to the video
        history : int
            determines how many frames influence the model
        detectShadows : boolean
            determines if shadowdetection is on or off
        """
        self.m=cv.createBackgroundSubtractorMOG2(history=history, detectShadows=False)
        self.video_path=video_path
        self.mask=None


    def begin_train(self, max_frames=500):
        """
        Train the model

        Parameters
        ----------
        max_frames : int
            determines with how many frames the model can be trained at most
        """

        # Read the video from specified path
        cam = cv.VideoCapture(self.video_path)
        frame_count=0
        while(True):
            # reading from frame
            ret,frame = cam.read()
            # if no video is left break or max frames is reached
            if frame is None or frame_count==max_frames:
                break
            # smooth
            image=cv.GaussianBlur(frame, ksize=(3,3),sigmaX=0)
            # update the background model
            self.m.apply(image)
            frame_count+=1
        # Release all space and windows once done
        cam.release()


    def getMask(self, image):
        """
        Calculates the foreground mask from given image

        Parameters
        ----------
        image :  np.array (n,m,3)

        Returns
        ----------
        self.mask : np.array (n,m) with values 0 or 1
        """
        image=cv.GaussianBlur(image, ksize=(3,3),sigmaX=0)
        mask = self.m.apply(image, learningRate = 0 )
        mask=mask==255
        mask=smooth_mask(mask)
        self.mask=mask
        return mask
    

    def getBackgroundImage(self):
        """
        Calculates the background image

        Returns
        ----------
        self.background : np.array (n,m,3)
        """
        self.background=self.m.getBackgroundImage()
        return self.background
    
    def train_on_image(self,image):
        image=cv.GaussianBlur(image, ksize=(3,3),sigmaX=0)
        self.m.apply(image, learningRate = -1 )

if __name__=="__main__":
    #toy_image=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t391.jpg"), cv.IMREAD_COLOR)
    #toy_image=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t244.jpg"), cv.IMREAD_COLOR)
    toy_image=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t193.jpg"), cv.IMREAD_COLOR)
    
    backsub=Backsub("Video_Data/take_02.avi",history=100)
    backsub.begin_train(max_frames=250)
    background=backsub.getBackgroundImage()
    mask=backsub.getMask(toy_image)

    foreground=np.copy(toy_image)
    foreground[mask==0]=0
    from hough_circle_trafo import hough_circle
    circles=hough_circle(foreground,draw=True,houghparameter=[1,40,80,10,15,19])

    cv.imshow("background",background)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow("toy_image",toy_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow("foreground",foreground)
    cv.waitKey(0)
    cv.destroyAllWindows()


