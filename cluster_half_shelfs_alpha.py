import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from low_pass import LowPass
from confectionery import Confectionery

class Half_shelf_cluster:
    """
    Intialize alf_shelf_cluster object and properties

    Parameters
    ----------
    object_trajectory: np.array(N,4,dtype=float) 
        object trajectory with (frame number, x coordinate, y coordinate, radius) respectively
    image_stream : list of len N
        strings of frame paths   
    confectionery: object of type Confectionery
    cut_radius: float or int
    """
    def __init__(self, object_trajectory, image_stream, confectionery, cut_radius):
        self.object_trajectory=object_trajectory
        self.image_stream=image_stream
        self.cut_radius=cut_radius
        self.confectionery=confectionery
        self.data=np.empty(0)
        self.current_index=-1
        self.cmap = { 0:'blue',1:'pink',2:'brown',3:'purple',4:'green'}

    def load_data_from_np_array(self, data):
        """
        Loads Data directly from data numpy array

        Parameters
        ----------
        data: np.array(M,dtype=np.uint8) 
            grayscale values stored in 1d array

        """
        self.data=data

    def load_data_initial(self, frame_portion=1.0):
        """
        Load data so we can cluster them

        Parameters
        ----------
        frame_portion : float between (0,1)
            portion of frames that will be used, standard value is 1.0 (all frames of object)
        Returns
        ----------
        current_index : int
            current index of frame
        """
        N=round((self.object_trajectory.shape[0])*frame_portion)
        for i in range(N):
            obj_image_current=cv.imread(self.image_stream[i], cv.IMREAD_GRAYSCALE)
            object_indeces=np.s_[round(self.object_trajectory[i,2]-round(self.object_trajectory[i,3])):round(self.object_trajectory[i,2]+round(self.object_trajectory[i,3]))+1,
                                round(self.object_trajectory[i,1]-round(self.object_trajectory[i,3])):round(self.object_trajectory[i,1]+round(self.object_trajectory[i,3]))+1]
            obj_image_current=obj_image_current[object_indeces]
            M=self.confectionery.sell_cake(radius=self.cut_radius,width=obj_image_current.shape[1],height=obj_image_current.shape[0])
            object_pixels=np.float32(obj_image_current[M==1])
            self.data=np.append(self.data,object_pixels)
        self.current_index=i
        return self.current_index
    
    def load_next_frame_into_data(self):
        """
        Load next frame data so we can cluster them

        Parameters
        ----------
        frame_portion : float between (0,1)
            portion of frames that will be used, standard value is 1.0 (all frames of object)
        Returns
        ----------
        current_index : int
            current index of frame
        """
        if self.current_index==len(self.image_stream)-1:
            return self.current_index
        self.current_index+=1
        obj_image_current=cv.imread(self.image_stream[self.current_index], cv.IMREAD_GRAYSCALE)
        object_indeces=np.s_[round(self.object_trajectory[self.current_index,2]-np.ceil(self.object_trajectory[self.current_index,3])):round(self.object_trajectory[self.current_index,2]+np.ceil(self.object_trajectory[self.current_index,3]))+1,
                            round(self.object_trajectory[self.current_index,1]-np.ceil(self.object_trajectory[self.current_index,3])):round(self.object_trajectory[self.current_index,1]+np.ceil(self.object_trajectory[self.current_index,3]))+1]
        obj_image_current=obj_image_current[object_indeces]
        cut_radius=round(np.floor(self.object_trajectory[self.current_index,-1]*self.cut_radius_portion))
        M=self.confectionery.sell_cake(radius=cut_radius,width=obj_image_current.shape[1],height=obj_image_current.shape[0])
        object_pixels=np.float32(obj_image_current[M==1])
        self.data=np.append(self.data,object_pixels)
        return self.current_index


    def train_k_means(self, k=2, plot_hist=False):
        """
        Train k means clustering model

        Parameters
        ----------
        k: int
            number of clusters
        plot_hist : boolean
            bool to decide if we plot or not
        """
        self.kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(self.data.reshape((-1,1)))
        self.labels_kmeans=self.kmeans.labels_
        self.centroids_kmeans=self.kmeans.cluster_centers_.reshape((-1))
        self.kmeans_idx=np.argsort(self.centroids_kmeans)
        self.centroids_kmeans[self.kmeans_idx]
        self.labels_kmeans=self.kmeans_idx[self.labels_kmeans]
        if plot_hist:
            plt.figure()
            n, bins, patches = plt.hist(self.data[self.labels_kmeans==0], bins=np.arange(min(self.data), max(self.data) +1, 1), color = "blue")
            d=np.abs(bins.reshape((-1,1))-self.centroids_kmeans.reshape((1,-1)))
            i1=np.argmin(d[:,0])
            i2=np.argmin(d[:,1])
            patches[i1].set_fc('r')
            patches[i2].set_fc('r')
            n, bins, patches = plt.hist(self.data[self.labels_kmeans==1], bins=np.arange(min(self.data), max(self.data) +1, 1), color = "green")
            patches[i1].set_fc('r')
            patches[i2].set_fc('r')
            plt.show()


    def train_gmm(self, number_of_gaussian=2, plot_hist=False):   
        """
        Trains gmm clustering model

        Parameters
        ----------
        number_of_gaussian: int
            number of gaussian the gmm will have
        plot_hist : boolean
            bool to decide if we plot or not
        """
        self.gmm = GaussianMixture(n_components=number_of_gaussian, random_state=0).fit(self.data.reshape((-1,1)))
        self.labels_gmm = self.gmm.fit_predict(self.data.reshape((-1,1)))
        self.priors_gmm = self.gmm.weights_
        self.means_gmm = self.gmm.means_.reshape((-1))
        self.variance = self.gmm.covariances_.reshape((-1))
        self.gmm_idx = np.argsort(self.means_gmm)
        self.means_gmm=self.means_gmm[self.gmm_idx]
        self.priors_gmm=self.priors_gmm[self.gmm_idx]
        self.variance=self.variance[self.gmm_idx]
        self.labels_gmm = self.gmm_idx[self.labels_gmm]

        if plot_hist:
            plt.figure()
            n, bins, patches = plt.hist(self.data, bins=np.arange(min(self.data), max(self.data) +1, 1), density=True)
            bin_labels=self.gmm_idx[self.gmm.predict(bins.reshape((-1,1)))]
            for l in range(bin_labels.size-1):
                label_index=bin_labels[l]
                patches[l].set_fc(self.cmap[label_index])
            for k in range(number_of_gaussian):
                d=np.abs(bins.reshape((-1,1))-self.means_gmm.reshape((1,-1)))
                for j in range(self.means_gmm.shape[0]):
                    i=np.argmin(d[:,j])
                    patches[i].set_fc('r')
                # plot gmm
                x = np.linspace(0,255, 1000)
                plt.plot(x, self.priors_gmm[k]*norm.pdf(x, self.means_gmm[k], np.sqrt(self.variance[k])))
            plt.show()
        return self.priors_gmm, self.gmm.score(self.data.reshape((-1,1)))



    def train_gmm_cutoff_filter(self, cut_off_probability=0.95, plot_hist=False):   
        """
        Trains gmm clustering model with cutoff filter

        Parameters
        ----------
        cut_off_probability: float
            cut off probability indicating correct fit to max cluster
        plot_hist : boolean
            bool to decide if we plot or not
        """
        X=np.copy(self.data.reshape((-1,1)))
        self.gmm_cutoff_filter = GaussianMixture(n_components=2, random_state=0).fit(X)
        prob=self.gmm_cutoff_filter.predict_proba(X)
        prob_post=np.max(prob, axis=1)
        X=X[prob_post>=cut_off_probability].reshape((-1,1))
        self.gmm_cutoff_filter = GaussianMixture(n_components=2, random_state=0).fit(X)
        self.labels_gmm_cutoff_filter = self.gmm_cutoff_filter.predict(self.data.reshape((-1,1)))
        self.priors_gmm_cutoff_filter = self.gmm_cutoff_filter.weights_
        self.means_gmm_cutoff_filter = self.gmm_cutoff_filter.means_.reshape((-1))
        self.variance_cutoff_filter = self.gmm_cutoff_filter.covariances_.reshape((-1))

        self.gmm_cutoff_filter_idx = np.argsort(self.means_gmm_cutoff_filter)
        self.means_gmm_cutoff_filter=self.means_gmm_cutoff_filter[self.gmm_cutoff_filter_idx]
        self.priors_gmm_cutoff_filter=self.priors_gmm_cutoff_filter[self.gmm_cutoff_filter_idx]
        self.variance_cutoff_filter=self.variance_cutoff_filter[self.gmm_cutoff_filter_idx]
        self.labels_gmm_cutoff_filter = self.gmm_cutoff_filter_idx[self.labels_gmm_cutoff_filter]

        if plot_hist:
            plt.figure()
            n, bins, patches = plt.hist(self.data, bins=np.arange(min(self.data), max(self.data) +1, 1), density=True)
            bin_labels=self.gmm_cutoff_filter_idx[self.gmm_cutoff_filter.predict(bins.reshape((-1,1)))]
            for l in range(bin_labels.size-1):
                label_index=bin_labels[l]
                patches[l].set_fc(self.cmap[label_index])
            for k in range(2):
                d=np.abs(bins.reshape((-1,1))-self.means_gmm_cutoff_filter.reshape((1,-1)))
                for j in range(self.means_gmm_cutoff_filter.shape[0]):
                    i=np.argmin(d[:,j])
                    patches[i].set_fc('r')
                # plot gmm
                x = np.linspace(0,255, 1000)
                plt.plot(x, self.priors_gmm_cutoff_filter[k]*norm.pdf(x, self.means_gmm_cutoff_filter[k], np.sqrt(self.variance_cutoff_filter[k])))
            plt.show()

    def label(self, data, method):
        """
        labels data

        Parameters
        ----------
        data: np.array(N,dtype=np.uint8)
            flat data
        method: string
            allowed are one of ["Kmeans","GMM","GMM_cutoff"]

        Returns
        -------
        labels: np.array(N,dtype=int)
            labels with values 0(sputtered) and 1(unsputtered)
        """
        if not(self.data.shape[0]):
            self.load_data_initial()
        if method=="Kmeans":
            if not(hasattr(self, 'kmeans')):
                self.train_k_means()
            labels=self.kmeans_idx[self.kmeans.predict(data.reshape((-1,1)))]
            return labels
        elif method=="GMM":
            if not(hasattr(self, 'gmm')):
                self.train_gmm()
            labels=self.gmm_idx[self.gmm.predict(data.reshape((-1,1)))]
            return labels
        elif method=="GMM_cutoff":
            if not(hasattr(self, 'gmm_cutoff_filter')):
                self.train_gmm_cutoff_filter()
            labels=self.gmm_cutoff_filter_idx[self.gmm_cutoff_filter.predict(data.reshape((-1,1)))]
            return labels
        else:
            raise Exception("Sorry, this method does not exist. Check yourn String!") 


    def train_gmm_ransac(self, cut_off_probability=0.85, initial_number_smaple_for_fitting=20, outlier_proportion=0.2 ,plot_hist=False):   
        """
        Trains gmm clustering model with ransac

        Parameters
        ----------
        cut_off_probability: float
            cut off probability indicating correct fit to max cluster
        plot_hist : boolean
            bool to decide if we plot or not
        """

        pass
    
    def get_priors_avg_log_likelihood(self):
        return self.priors_gmm, self.gmm.score(self.data.reshape((-1,1)))
                

if __name__=="__main__":
    # load all necessary data
    object_trajectory=my_data = np.genfromtxt('/Users/Hoang_1/Desktop/Master_Arbeit/software/janus_particle_tracking/object_0_trajectory.csv', delimiter=',')
    object_trajectory=object_trajectory[1:,:]
    stream_string=[]
    koppenrath=Confectionery()
    for i in range(object_trajectory.shape[0]):
        frame_number=round(object_trajectory[i,0])
        im_path="Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t"+str(frame_number).zfill(3)+".jpg"
        stream_string.append(im_path)
    # cluster
    half_shelf_cluster=Half_shelf_cluster(object_trajectory, stream_string, confectionery=koppenrath, cut_radius=11)
    half_shelf_cluster.load_data_initial(frame_portion=1.0)
    half_shelf_cluster.train_2_means(plot_hist=True)
    half_shelf_cluster.train_gmm(number_of_gaussian=3, plot_hist=True)
    half_shelf_cluster.train_gmm_cutoff_filter(plot_hist=True)



