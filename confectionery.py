from cake import cake
import numpy as np
from matplotlib import pyplot as plt

class Confectionery:
    """
    Confectionery class, stores cakes and donuts
    """
    def __init__(self):
        self.cake_fridge=dict()
        self.cake_fridge[0]=np.ones((1,1), dtype=np.bool_)
        self.donut_fridge=dict()
        self.donauwelle_fridge=dict()
        self.funnelcake_fridge=dict()

    def bake_cake(self,radius):
        M=cake(radius)
        self.cake_fridge[radius]=M
        return M
    
    def sell_cake(self, radius, width, height):
        if radius in self.cake_fridge:
            M=self.cake_fridge[radius]
        else:
            M=self.bake_cake(radius)
        x_width=(width-M.shape[1])//2
        y_width=(height-M.shape[0])//2
        M=np.pad(M,[(y_width,y_width),(x_width,x_width)],mode="constant")
        return M
    
    def bake_donut(self,radius):
        """
        bakes a donut. Donut is a 2D matrix consisting of zeros and ones, 
        where pixels with value one have pixles_distances between (radius-1,radius]
        """
        if radius==0:
            return self.bake_cake(radius)
        if not(radius in self.cake_fridge):
            self.bake_cake(radius)
        if not((radius-1) in self.cake_fridge):
            self.bake_cake(radius)
        M1=self.sell_cake(radius, 2*radius+1, 2*radius+1)
        M2=self.sell_cake(radius-1, 2*radius+1, 2*radius+1)
        M=np.logical_and(M1,1-M2)
        self.donut_fridge[radius]=M
        return M
    
    def sell_donut(self,radius, width, height):
        if radius in self.donut_fridge:
            M=self.donut_fridge[radius]
        else:
            M=self.bake_donut(radius)
        x_width=(width-M.shape[1])//2
        y_width=(height-M.shape[0])//2
        M=np.pad(M,[(y_width,y_width),(x_width,x_width)],mode="constant")
        return M
    
    def bake_donauwelle(self, half_width):
        """
        we only allow square shaped donauwellen, since they are better!
        """
        X=np.arange(-half_width,half_width+1)
        X,_=np.meshgrid(X,X)
        self.donauwelle_fridge[half_width]=X
        return X
    
    def sell_donauwelle(self, half_width):
        if half_width in self.donauwelle_fridge:
            M=self.donauwelle_fridge[half_width]
        else:
            M=self.bake_donauwelle(half_width)
        return M
    
    def bake_funnelcake(self, radius):
        """
        shape is hard to describe, it is actually related to general cake baking
        values are np.sqrt(x**2+y**2)
        """
        X=self.sell_donauwelle(radius)
        Y=np.copy(np.transpose(X))
        R=np.sqrt(X**2+Y**2)
        self.funnelcake_fridge[radius]=R
        return R
    
    def sell_funnelcake(self, radius):
        if radius in self.funnelcake_fridge:
            R=self.funnelcake_fridge[radius]
        else:
            R=self.bake_funnelcake(radius)
        return R
    
if __name__=="__main__":
    max_radius=5
    height=2*max_radius+1
    width=2*max_radius+1
    cakes=np.zeros((height,width,max_radius+1))
    donuts=np.zeros((height,width,max_radius))

    koppenrath=Confectionery()
    for i in range(max_radius+1):
        cakes[:,:,i]=koppenrath.sell_cake(i,width=width,height=height)
        if i!=0:
            donuts[:,:,i-1]=koppenrath.sell_donut(i,width=width,height=height)
    plt.figure()
    plt.subplot(max_radius+1,2,1)
    for i in range(max_radius+1):
        if i==0:
            plt.subplot(max_radius+1,2,1)
            plt.imshow(cakes[:,:,i])
            plt.subplot(max_radius+1,2,2)
            plt.imshow(cakes[:,:,i])
        else:
            plt.subplot(max_radius+1,2,2*i+1)
            plt.imshow(cakes[:,:,i])
            plt.subplot(max_radius+1,2,2*i+2)
            plt.imshow(donuts[:,:,i-1])
    plt.show()
