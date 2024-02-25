from hough_circle_trafo import hough_circle
from cake import cake
from background_subtraction_gmm import Backsub
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class Segmentation:
    """
    Segmentation class,
    Clusters pxiel into object and background
    """
    def __init__(self, backsub):
        """
        Initialize Segmentation class

        Parameters
        ----------
            backsub : trained backsub object
        """
        self.backsub=backsub

    def segment_image(self,image, dp=1, min_dist=40, canny_upper=80, canny_lower=10, minradius=5, maxradius=None):
        """
        Segments the input image with Hough Circle and Background Substraction

        Parameters
        ----------
            image: np.array
                input image in bgr 
        
        Returns
        ----------
            self.circles: Nx3 np.array,
                where N is number of circles and 2nd dimension is x center and y center and radius
                if no object detected returns None
        """
        self.image=image
        mask=self.backsub.getMask(self.image)
        self.foreground_mask=mask
        contours, _ =cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        num_roi=len(contours)
        if num_roi==0:
            return None
        rois=[]
        for i in range(num_roi):
            x_min=np.min(contours[i][:,0,0])
            y_min=np.min(contours[i][:,0,1])
            x_max=np.max(contours[i][:,0,0])
            y_max=np.max(contours[i][:,0,1])
            x_center=(x_max+x_min)//2
            y_center=(y_max+y_min)//2
            if (x_max-x_min)%2==1:
                # odd
                w1=(x_max-x_min)//2
                w2=w1+1
            else:
                # even
                w1=(x_max-x_min)//2
                w2=w1
            if (y_max-y_min)%2==1:
                # odd
                h1=(y_max-y_min)//2
                h2=h1+1
            else:
                # even
                h1=(y_max-y_min)//2
                h2=h1
            # double the backround range
            w1=2*w1
            w2=2*w2
            h1=2*h1
            h2=2*h2
            y_min_cropped=np.max([0,y_center-h1])
            y_max_cropped=np.min([self.image.shape[0],y_center+h2+1])
            x_min_cropped=np.max([0,x_center-w1])
            x_max_cropped=np.min([self.image.shape[1],x_center+w2+1])
            roi=[np.copy(self.image[y_min_cropped:y_max_cropped,x_min_cropped:x_max_cropped,:]),np.array([x_min_cropped,y_min_cropped])]
            rois.append(roi)
        circles=np.empty((0,3))
        for roi in rois:
            r_max=np.min(roi[0].shape[:1])//2
            r_max=np.max([r_max,minradius])
            if maxradius is not None:
                r_max=maxradius
            circle=hough_circle(roi[0], maxradius=r_max, dp=dp, min_dist=min_dist, canny_upper=canny_upper, canny_lower=canny_lower, minradius=minradius)
            if type(circle) is np.ndarray:
                circle[:,:2]=circle[:,:2]+roi[1].reshape((1,2))
                circles=np.concatenate((circles,circle),axis=0)
        if circles.shape[0]==0:
            return None
        self.circles=circles
        """
        self.circles=hough_circle(image*self.foreground_mask[:,:,np.newaxis])
        #self.circles=hough_circle(image)
        if type(self.circles) is np.ndarray:
            true_circle_index=np.ones((self.circles.shape[0]), dtype=bool)
            for i in range(true_circle_index.shape[0]):
                radius=round(self.circles[i,-1])
                x_pos=round(self.circles[i,0])
                y_pos=round(self.circles[i,1])
                M=cake(radius)
                object_foreground=self.foreground_mask.copy()
                object_foreground=np.pad(object_foreground, pad_width=[(radius+1, radius+1),(radius+1, radius+1)], mode='constant')
                object_foreground=object_foreground[y_pos+1:y_pos+2*radius+2,x_pos+1:x_pos+2*radius+2]
                if np.sum(object_foreground*M)<=np.sum(M)*0.8:
                    true_circle_index[i]=False
        self.circles=self.circles[true_circle_index,:]
        """
        """
        self.binary_mask=np.zeros(self.foreground_mask.shape,np.uint8)
        # pad binary massk with 0 with half of max circle radius
        max_circle_radius=20
        self.binary_mask=np.pad(self.binary_mask, pad_width=[(max_circle_radius, max_circle_radius),(max_circle_radius, max_circle_radius)], mode='constant')
        if type(self.circles) is np.ndarray:
            for i in range(self.circles.shape[0]):
                M=cake(self.circles[i,-1])
                window_length=M.shape[0]//2
                y=round(self.circles[i,1])
                x=round(self.circles[i,0])
                self.binary_mask[max_circle_radius+y-window_length:max_circle_radius+y+window_length+1,max_circle_radius+x-window_length:max_circle_radius+x+window_length+1]=M.astype(np.uint8)
        # undo padding
        self.binary_mask=self.binary_mask[max_circle_radius:-max_circle_radius,max_circle_radius:-max_circle_radius]
        """
        return self.circles
    

    def get_object_list(self):
        """
        method to cluster into object

        return : list with np.arrays
            lenght of list are number of objects, 
            each np.array have the y and x coordinates of the pixels belonging to the object

        """
        contours, hierarchy = cv.findContours(self.binary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.objects=[]
        mass_center=np.zeros((self.circles.shape[0],2))
        for i in range(len(contours)):
            cimg = np.zeros_like(self.binary_mask)
            cv.drawContours(cimg, contours, i, color=255, thickness=-1)
            py, px = np.where(cimg == 255)
            M = cv.moments(contours[i])
            mass_center[i,:]=np.array([M['m10']/M['m00'],M['m01']/M['m00']])
            self.objects.append(np.vstack((py,px)))
        permutation=[]
        for i in range(self.circles.shape[0]):
            dist=mass_center-self.circles[i,0:2]
            dist=np.sum(dist**2,1)
            permutation.append(np.argmin(dist))
        return [self.objects[j] for j in permutation]
    
    def get_object_images(self):
        """
        returns list of object_images (ROI)
        """
        self.object_images=[]
        for i in range(self.circles.shape[0]):
            x=round(self.circles[i,0])
            y=round(self.circles[i,1])
            r=round(self.circles[i,2])
            a=5
            self.object_images.append(self.image[y-r-a:y+r+a+1,x-r-a:x+r+a+1,:])
        return self.object_images


def main(image):
    backsub=Backsub("Video_Data/take_02.avi", history=100)
    backsub.begin_train(max_frames=250)
    segm=Segmentation(backsub=backsub)
    segm.segment_image(image)
    return segm


if __name__=="__main__":
    #src=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t011.jpg"), cv.IMREAD_COLOR)
    #src=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t391.jpg"), cv.IMREAD_COLOR)
    #src=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t494.jpg"), cv.IMREAD_COLOR)
    #src=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t244.jpg"), cv.IMREAD_COLOR)
    src=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t193.jpg"), cv.IMREAD_COLOR)
    #src=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t222.jpg"), cv.IMREAD_COLOR)

    segm=main(src)
    image=np.copy(src)

    for i in segm.circles.astype(int):
        center = (i[0], i[1])
        # circle outline
        radius = i[2]
        # circle center
        cv.circle(image, center, 1, (0, 100, 100), 3)
        cv.circle(image, center, radius, (255, 0, 255), 3)

    cv.imshow("detected circles",image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow("mask", 255*segm.binary_mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    objects=segm.get_object_list()
    mask=np.zeros(src.shape[0:2],np.uint8)
    k=1
    for i in range(len(objects)):
        mask[objects[i][0][:],objects[i][1][:]]=k
        k+=1
    blue_vector=np.reshape(np.array([255,0,0]),(1,1,3))
    green_vector=np.reshape(np.array([0,128,0]),(1,1,3))
    purple_vector=np.reshape(np.array([221,160,221]),(1,1,3))

    image_clustered=np.zeros(src.shape,np.uint8)
    image_clustered[mask==1,:]=blue_vector
    image_clustered[mask==2,:]=green_vector
    image_clustered[mask==3,:]=purple_vector

    cv.imshow("clustered objects", image_clustered)
    cv.waitKey(0)
    cv.destroyAllWindows()

    object_images=segm.get_object_images()
    cv.imshow("object 1", object_images[0])
    cv.waitKey(0)
    cv.destroyAllWindows()

    plt.figure()
    plt.subplot(1,3,1)
    plt.title("Original Image Frame 244")
    plt.imshow(src)
    plt.subplot(1,3,2)
    plt.title("Segmented Image Frame 244")
    plt.imshow(image_clustered)

    src=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t245.jpg"), cv.IMREAD_COLOR)
    segm=main(src)
    circle=segm.circles[0].astype(int)
    center = (circle[0], circle[1])
    # circle outline
    radius = circle[2]
    # circle center
    cv.circle(src, center, 1, (0, 100, 100), 3)
    cv.circle(src, center, radius, (255, 0, 255), 3)
    plt.subplot(1,3,3)
    plt.title("Original Image Frame 245 with highlighted object")
    plt.imshow(src)
    plt.show()


