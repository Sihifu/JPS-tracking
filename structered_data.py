import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from low_pass import LowPass
from confectionery import Confectionery
from scipy import ndimage

class StructuredData:
    """
    Intialize StructuredData

    Parameters
    ----------
    object_trajectory: np.array(N,4,dtype=float) 
        object trajectory with (frame number, x coordinate, y coordinate, radius) respectively
    image_stream : list of len N
        strings of frame paths   
    confectionery: object of type Confectionery
    """
    def __init__(self, object_trajectory, image_stream, confectionery):
        self.object_trajectory=object_trajectory
        self.image_stream=image_stream
        self.confectionery=confectionery
        self.r_estimate=np.mean(object_trajectory[:,-1])
        self.M=self.confectionery.sell_cake(round(self.r_estimate),2*round(self.r_estimate)+1,2*round(self.r_estimate)+1)
        self.time_indeces=np.flatnonzero(self.M)

    def create_data(self):
        """
        creates the data structure

        Returns
        -------
        data: np.array(N,M,M,dtype=np.uint8) with N the amount of frames and M the max estiamted radius
            data structure where 1 dim is temporal and the other dimension are spatial
        """
        object_indeces=[np.s_[round(y-round(self.r_estimate)):round(y+round(self.r_estimate))+1,
                     round(x-round(self.r_estimate)):round(x+round(self.r_estimate))+1] for x,y in zip(self.object_trajectory[:,1],self.object_trajectory[:,2])]
        ims=[cv.imread(l, cv.IMREAD_GRAYSCALE) for l in self.image_stream]
        M=np.repeat(self.M[np.newaxis,:,:],len(ims),axis=0)
        data=np.array([a[b] for a,b in zip(ims,object_indeces)])
        data[M==0]=0
        self.data=data
        return data

    def load_data(self,path):
        self.data=np.load(path)
        return self.data

    def save_data(self,name,path):
        np.save(path+name,self.data)

    def spatial_filter_data(self,filter,mode='constant',cval=0.0):
        data_filtered=np.array([ndimage.convolve(a.astype(np.float32), filter, mode=mode, cval=cval) for a in self.data])
        return data_filtered

    def temporal_filter_data(self,filter, mode='constant', cval=0.0):
        time_data=self.data.reshape((self.data.shape[0],-1))
        output=np.copy(time_data).astype(np.float32)
        time_data=time_data[:,self.time_indeces]
        data_filtered=ndimage.convolve1d(time_data.astype(np.float32), filter, mode=mode, cval=cval, axis=0)
        output[:,self.time_indeces]=data_filtered
        output=output.reshape(self.data.shape)
        return output

    def overwrite_data(self,data):
        self.data=data
        return data

    def cut_data(self,cut_radius):
        x=self.data.shape[1]//2
        self.data=self.data[:,x-cut_radius:x+cut_radius+1,x-cut_radius:x+cut_radius+1]
        self.M=self.confectionery.sell_cake(cut_radius,self.data.shape[1],self.data.shape[2])
        self.time_indeces=np.flatnonzero(self.M)
        M=np.repeat(self.M[np.newaxis,:,:],self.data.shape[0],axis=0)
        self.data[M==0]=0
        return self.data


if __name__=="__main__":
    import numpy as np
    from confectionery import Confectionery
    obj_0=np.load('data/obj_0_traj.npy')
    start_frame=round(obj_0[0,0])
    end_frame=round(obj_0[-1,0])
    stream_string=[]
    for i in range(start_frame,end_frame+1):
        im_path="/Users/Hoang_1/Desktop/Master_Arbeit/software/janus_particle_tracking/Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t"+str(i).zfill(3)+".jpg"
        stream_string.append(im_path)
    koppenrath=Confectionery()
    structered_data=StructuredData(obj_0,stream_string,koppenrath)
    structered_data.create_data()
    structered_data.save_data(name="data_structured.npy",path="data/")




