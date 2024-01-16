import numpy as np
from scipy import ndimage
import cv2 as cv
from confectionery import Confectionery
from matplotlib import pyplot as plt

class RadialDerivative:
    def __init__(self, gray, confectionery):
        self.gray=gray.astype(np.float32)
        self.gray_x=ndimage.sobel(self.gray, 1, mode="constant") 
        self.gray_y=ndimage.sobel(self.gray, 0, mode="constant")
        self.confectionery=confectionery

    def calculate_derivative_magnitude(self, valid=False):
        self.deriv_magn=np.sqrt(self.gray_x**2+self.gray_y**2)
        if valid:
            self.deriv_magn=self.deriv_magn[1:-1,1:-1]
        return self.deriv_magn
    
    def calculate_derivative_squared(self, valid=False):
        self.deriv_sq=self.gray_x**2+self.gray_y**2
        if valid:
            self.deriv_sq=self.deriv_sq[1:-1,1:-1]
        return self.deriv_sq

    def calculate_derivative(self, valid=False):
        half_width=(self.gray.shape[1]-1)//2
        X=self.confectionery.sell_donauwelle(half_width)
        Y=np.copy(np.transpose(X))
        R=np.copy(self.confectionery.sell_funnelcake(half_width))
        # account for non differentiability of r=0
        R[half_width,half_width]=1
        self.radial_derivative=(X*self.gray_x+Y*self.gray_y)/R
        a=self.radial_derivative[half_width-1:half_width+2,half_width-1:half_width+2].reshape(-1)
        b=(1/8)*np.ones((9))
        b[4]=0
        self.radial_derivative[half_width,half_width]=np.sum(a*b)
        if valid:
            self.radial_derivative=self.radial_derivative[1:-1,1:-1]
        return self.radial_derivative

if __name__=="__main__":
    koppenrath=Confectionery()
    radius=10
    M=koppenrath.sell_cake(radius,2*radius+1,2*radius+1)
    im=np.arange(1,4).reshape((1,3))
    im=np.vstack((im,im))
    im=np.vstack((im,im[0,:].reshape((1,3))))
    r=RadialDerivative(M, koppenrath)
    dr=r.calculate_derivative()
    plt.figure()
    plt.imshow(dr)
    plt.colorbar()
    plt.figure()
    plt.imshow(r.gray_x)
    plt.colorbar()
    plt.figure()
    plt.imshow(np.sqrt(r.gray_x**2+r.gray_y**2))
    plt.colorbar()
    plt.show()
    
    
