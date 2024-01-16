import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from low_pass import LowPass
from confectionery import Confectionery
from scipy import ndimage

class ParticleDataStruct:
    """
    Intialize TestDataStruct

    Parameters
    ----------
    object_trajectory: np.array(N,4,dtype=float) 
        object trajectory with (frame number, x coordinate, y coordinate, radius) respectively
    image_stream : list of len N
        strings of frame paths   
    confectionery: object of type Confectionery
    r_estimate: int, estimated radius for inner clustering
    """
    def __init__(self, object_trajectory, image_stream, r_estimate, confectionery):
        self.object_trajectory=object_trajectory
        self.image_stream=image_stream
        self.confectionery=confectionery
        self.r_estimate=r_estimate
        self.M=self.confectionery.sell_cake(round(self.r_estimate),2*round(self.r_estimate)+1,2*round(self.r_estimate)+1)
        self.ind_pos=np.flatnonzero(self.M)

    def create_data_3d(self):
        """
        creates the data structure

        Returns
        -------
        data: np.array(N,M,M,dtype=np.uint8) with N the amount of frames and M the max estiamted radius
            data structure where 1 dim is temporal and the other dimension are spatial
        """
        object_indeces=[np.s_[round(y-round(self.r_estimate)):round(y+round(self.r_estimate))+1,
                     round(x-round(self.r_estimate)):round(x+round(self.r_estimate))+1] for x,y in zip(self.object_trajectory[:,1],self.object_trajectory[:,2])]
        start_frame=round(self.object_trajectory[0,0])
        end_frame=round(self.object_trajectory[-1,0])
        ims=[cv.cvtColor(np.load(l),cv.COLOR_BGR2GRAY) for l in self.image_stream[start_frame:end_frame+1]]
        M=np.repeat(self.M[np.newaxis,:,:],len(ims),axis=0)
        data=np.array([a[b] for a,b in zip(ims,object_indeces)])
        data[M==0]=0
        self.data_3d=data
        return data

    def load_data_3d(self,path):
        self.data_3d=np.load(path)
        return self.data_3d

    def save_data_3d(self,path):
        np.save(path,self.data_3d)

    def flatten_data(self):
        data_flat=self.data_3d.reshape((self.data_3d.shape[0],-1))
        data_flat=data_flat[:,self.ind_pos]
        self.data_flat=data_flat
        return data_flat
    
    def load_data_flat(self,path):
        self.data_flat=np.load(path)
        return self.data_flat

    def save_data_flat(self,path):
        np.save(path,self.data_flat)
    
    def get_data_flatten_effective(self,cut_radius,data_flat):
        M=self.confectionery.sell_cake(cut_radius,2*self.r_estimate+1,2*self.r_estimate+1)
        ind_pos_effective=np.flatnonzero(M)
        ind_pos_effective=np.in1d(self.ind_pos,ind_pos_effective)
        data_flat_effective=data_flat[:,ind_pos_effective]
        return data_flat_effective

    def data_flatten_effective_to_data_flat(self,cut_radius,data_flat_effective):
        data_flat=np.zeros((data_flat_effective.shape[0],self.ind_pos.size))
        M=self.confectionery.sell_cake(cut_radius,2*self.r_estimate+1,2*self.r_estimate+1)
        ind_pos_effective=np.flatnonzero(M)
        ind_pos_effective=np.in1d(self.ind_pos,ind_pos_effective)
        data_flat[:,ind_pos_effective]=data_flat_effective
        return data_flat

    def sort_flat_array(self):
        perm=np.zeros(0,dtype=int)
        for r in range(round(self.r_estimate)+1):
            if r==0:
                M=self.confectionery.sell_cake(r,self.M.shape[1],self.M.shape[0])
            else:
                M=self.confectionery.sell_donut(r,self.M.shape[1],self.M.shape[0])
            ind_pos=np.flatnonzero(M)
            perm=np.append(perm,ind_pos)

    def expand_data(self,data_flat):
        expanded_data=np.zeros((data_flat.shape[0],2*self.r_estimate+1,2*self.r_estimate+1))
        expanded_data=expanded_data.reshape(((data_flat.shape[0],-1)))
        expanded_data[:,self.ind_pos]=data_flat
        expanded_data=expanded_data.reshape((data_flat.shape[0],2*self.r_estimate+1,2*self.r_estimate+1))
        return expanded_data






