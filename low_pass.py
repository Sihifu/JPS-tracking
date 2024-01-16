import numpy.fft as nf
import numpy as np 
import cv2 as cv
from cake import cake
from segmentation import Segmentation as Segmentation
from background_subtraction_gmm import Backsub as Backsub
from matplotlib import pyplot as plt
import scipy.signal as signal
from matplotlib import cm
from confectionery import Confectionery
from radial_derivative import RadialDerivative
class LowPass:
    """
    Class to filter an object, with different methods

    Parameters
    ----------
    object_image: np.array((m,m), dtype=np.uint8) m is odd integer
        grayscale of image of object
    cut_portion: float between (0,1)
        cutting portion if we want to cut circular object
    normalized_cut_off_frequency: float
        normalized cut off fequency for low pass
    """
    def __init__(self, confectionery, normalized_cut_off_frequency=8.0, delta_radius=1):
        self.delta_radius=delta_radius
        self.confectionery=confectionery
        self.normalized_cut_off_frequency=normalized_cut_off_frequency
        self.object_image_fourier=None

    def read_obj_image(self, object_image, with_FFT=True):
        self.object_image=np.copy(object_image).astype(np.float32)
        if self.delta_radius!=0:
            radius=((self.object_image.shape[0]-1)//2)-self.delta_radius
            self.object_image=self.cut_cake(radius=radius)
        if with_FFT:
            self.object_image_fourier=nf.fftshift(nf.fft2(self.object_image))

    def ideal_lowpass(self):
        M=self.confectionery.sell_cake(self.normalized_cut_off_frequency, self.object_image.shape[0], self.object_image.shape[0])
        filtered_image=np.copy(self.object_image_fourier)
        filtered_image[M==0]=0
        self.filtered_image=np.abs(nf.ifft2(nf.ifftshift(filtered_image))).astype(np.float32)
        return self.filtered_image

    def butter_worth_lowpass(self, order=4):
        m=self.object_image_fourier.shape[0]
        D=np.arange(-np.ceil(m//2),np.ceil(m//2)+1)**2
        D = D[:,None] + D
        D=np.sqrt(D)
        H = 1/(1+(D/self.normalized_cut_off_frequency)**(2*order))
        filtered_image_fourier=H*self.object_image_fourier
        self.filtered_image=np.abs(nf.ifft2(nf.ifftshift(filtered_image_fourier))).astype(np.float32)
        return self.filtered_image
    
    def cut_cake(self, radius=None, keep_dim=False):
        if not(radius is None):
            cut_radius=radius
        else:
            cut_radius=self.object_image.shape[0]//2
        self.M=self.confectionery.sell_cake(cut_radius, self.object_image.shape[0], self.object_image.shape[0])
        self.filtered_image=np.copy(self.object_image)
        self.filtered_image[self.M==0]=0
        pad_width=(self.object_image.shape[0]-2*cut_radius-1)//2
        if pad_width!=0 and not(keep_dim):
            self.filtered_image=self.filtered_image[pad_width:-pad_width,pad_width:-pad_width]
            self.M=self.M[pad_width:-pad_width,pad_width:-pad_width]
        return self.filtered_image
    
    def derivative_filter(self, object_image):
        self.read_obj_image(object_image, with_FFT=True)
        self.filtered_image=self.butter_worth_lowpass()
        self.read_obj_image(self.filtered_image, with_FFT=False)
        object_image_normalized=self.normalize()
        self.filtered_image=object_image_normalized
        obj_derivative=RadialDerivative(self.filtered_image, self.confectionery)
        obj_derivative=obj_derivative.calculate_derivative(valid=True)
        return obj_derivative

    def current_filter(self, object_image, delta_raidus=1):
        self.delta_radius=delta_raidus
        self.read_obj_image(object_image, with_FFT=True)
        self.filtered_image=self.butter_worth_lowpass()
        self.delta_radius=0
        self.read_obj_image(self.filtered_image, with_FFT=False)
        self.filtered_image=self.cut_cake((self.filtered_image.shape[0]-1)//2)
        return self.filtered_image

    def normalize(self):
        #self.object_image_normalized=(self.object_image-np.min(self.object_image))/(np.max(self.object_image)-np.min(self.object_image))
        radius=(self.object_image.shape[0]-1)//2
        M=self.confectionery.sell_cake(radius,self.object_image.shape[0],self.object_image.shape[0])
        mean=np.mean(self.object_image[M==1])
        variance=np.sum(((self.object_image[M==1]-mean)**2))/(np.sum(M==1)-1)
        self.object_image_normalized=(self.object_image-mean)/np.sqrt(variance)
        self.object_image_normalized[M==0]=0
        return self.object_image_normalized

        
if __name__=="__main__":
    frame_number=30
    normalized_cut_off_frequency=8.0
    delta_radius=1
    delta_cut=3
    object_trajectory=my_data = np.genfromtxt('/Users/Hoang_1/Desktop/Master_Arbeit/software/janus_particle_tracking/object_0_trajectory.csv', delimiter=',')
    object_trajectory=object_trajectory[1:,:]
    start_frame=round(object_trajectory[0,0])
    end_frame=round(object_trajectory[-1,0])
    stream_string=[]
    for i in range(start_frame,end_frame+1):
        im_path="Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t"+str(i).zfill(3)+".jpg"
        stream_string.append(im_path)
    obj_image_current=cv.imread(stream_string[frame_number], cv.IMREAD_GRAYSCALE)   
    circle=object_trajectory[frame_number,:] 
    y_pos=object_trajectory[frame_number,2]
    x_pos=object_trajectory[frame_number,1]
    radius=round(object_trajectory[frame_number,3])
    object_indeces=np.s_[round(y_pos-radius):round(y_pos+radius)+1,
                        round(x_pos-radius):round(x_pos+radius)+1]
    obj_image_current=obj_image_current[object_indeces]
    koppenrath=Confectionery()
    lp=LowPass(confectionery=koppenrath ,normalized_cut_off_frequency=normalized_cut_off_frequency, delta_radius=delta_radius)
    lp.read_obj_image(obj_image_current)
    obj_image_current=lp.cut_cake()
    lp.read_obj_image(obj_image_current)
    image_filtered_1=lp.ideal_lowpass()
    image_filtered_2=lp.butter_worth_lowpass(order=2)
    cut_radius=(obj_image_current.shape[0]-1)//2 - delta_cut
    image_filtered_3=lp.cut_cake(cut_radius,keep_dim=True)
  
    x_center=round(circle[0])
    y_center=round(circle[1])
    radius=obj_image_current.shape[0]//2
    x=np.arange(2*radius+1)+x_center
    y=np.arange(2*radius+1)+y_center
    y=y[::-1]
    X, Y = np.meshgrid(x, y)

    fig=plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_surface(X, Y, obj_image_current,cmap=cm.magma,linewidth=0,antialiased=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.title("Heatmap without lowpass filter")

    radius=image_filtered_1.shape[0]//2
    x=np.arange(2*radius+1)+x_center
    y=np.arange(2*radius+1)+y_center
    y=y[::-1]
    X, Y = np.meshgrid(x, y)
    ax = fig.add_subplot(2, 2, 2, projection='3d')  
    ax.plot_surface(X, Y, image_filtered_1,cmap=cm.magma,linewidth=0,antialiased=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.title("Heatmap with ideal lowpass filter")

    radius=image_filtered_2.shape[0]//2
    x=np.arange(2*radius+1)+x_center
    y=np.arange(2*radius+1)+y_center
    y=y[::-1]
    X, Y = np.meshgrid(x, y)
    ax = fig.add_subplot(2, 2, 3, projection='3d')  
    ax.plot_surface(X, Y, image_filtered_2,cmap=cm.magma,linewidth=0,antialiased=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.title("Heatmap with butterworth lowpass filter of order 2")

    radius=image_filtered_3.shape[0]//2
    x=np.arange(2*radius+1)+x_center
    y=np.arange(2*radius+1)+y_center
    y=y[::-1]
    X, Y = np.meshgrid(x, y)
    ax = fig.add_subplot(2, 2, 4, projection='3d')  
    ax.plot_surface(X, Y, image_filtered_3,cmap=cm.magma,linewidth=0,antialiased=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.title("Heatmap with radial cut off filter")

    radius=obj_image_current.shape[0]//2
    x=np.arange(2*radius+1)+x_center
    y=np.arange(2*radius+1)+y_center
    y=y[::-1]
    X, Y = np.meshgrid(x, y)
    lp=LowPass(confectionery=koppenrath)
    image_filtered_final=lp.current_filter(obj_image_current, delta_raidus=1)
    fig=plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, obj_image_current,cmap=cm.magma,linewidth=0,antialiased=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.title("Heatmap without lowpass filter")

    x_center=round(circle[0])
    y_center=round(circle[1])
    radius=image_filtered_final.shape[0]//2
    x=np.arange(2*radius+1)+x_center
    y=np.arange(2*radius+1)+y_center
    y=y[::-1]
    X, Y = np.meshgrid(x, y)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, image_filtered_final,cmap=cm.magma,linewidth=0,antialiased=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.title("Heatmap with final filter")


    r_effective=10
    #r_effective=(obj_image_current_alt.shape[0]-1)//2
    M=koppenrath.sell_cake(r_effective,obj_image_current.shape[0],obj_image_current.shape[0])
    X_org=np.copy(obj_image_current).astype(np.float32)
    X_org=X_org[M==1]
    M=koppenrath.sell_cake(r_effective,image_filtered_1.shape[0],image_filtered_1.shape[0])
    X1=np.copy(image_filtered_1).astype(np.float32)
    X1=X1[M==1]
    X2=np.copy(image_filtered_2).astype(np.float32)
    X2=X2[M==1]
    X3=np.copy(image_filtered_3).astype(np.float32)
    X3=X3[M==1]
    error1=np.mean(np.abs(X1-X_org))
    error2=np.mean(np.abs(X2-X_org))
    error3=np.mean(np.abs(X3-X_org))
    plt.figure()
    plt.scatter(np.arange(1,4),[error1, error2, error3])
    plt.title("error for effective radius " + str(r_effective))
    plt.show()
