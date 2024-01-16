import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from confectionery import Confectionery
from low_pass import LowPass
from radial_derivative import RadialDerivative

class Half_shelf_cluster:
    def __init__(self, object_trajectory, image_stream, confectionery, lowpass):
        """
        Intialize alf_shelf_cluster object and properties

        Parameters
        ----------
        object_trajectory: np.array Nx4 (frame number, x coordinate, y coordinate, radius)
        image_stream : list of len N
            strings of frame paths    
        """
        self.object_trajectory=object_trajectory
        self.image_stream=image_stream
        self.lowpass=lowpass
        self.confectionery=confectionery
        self.data=np.empty(0)
        self.derivatives=dict()
        self.rmax=0
        self.current_index=-1
        self.additive_image=None
        self.object_indeces=None
        self.load_all_object_indeces()
        self.object_tensor=None
        self.load_object_tensor()
        self.cmap = { 0:'blue',1:'pink',2:'brown',3:'purple',4:'green'}

    def load_all_object_indeces(self):
        self.object_indeces=[]
        for i in range(len(self.image_stream)):
            a=np.s_[round(self.object_trajectory[i,2]-round(self.object_trajectory[i,3])):round(self.object_trajectory[i,2]+round(self.object_trajectory[i,3]))+1,
                                round(self.object_trajectory[i,1]-round(self.object_trajectory[i,3])):round(self.object_trajectory[i,1]+round(self.object_trajectory[i,3]))+1]
            self.object_indeces.append(a)

    def load_object_indeces(self, object_trajectory, image_stream):
        for i in range(len(image_stream)):
            a=np.s_[round(object_trajectory[i,2]-round(object_trajectory[i,3])):round(object_trajectory[i,2]+round(object_trajectory[i,3]))+1,
                                round(object_trajectory[i,1]-round(object_trajectory[i,3])):round(object_trajectory[i,1]+round(object_trajectory[i,3]))+1]
            self.object_indeces.append(a)

    def load_object_tensor(self):
        self.rmax=round(np.ceil(np.max(self.object_trajectory[:,-1])))
        self.object_tensor=np.zeros((2*self.rmax+1,2*self.rmax+1,len(self.image_stream)))
        for i in range(len(self.image_stream)):
            obj_image_current=cv.imread(self.image_stream[i], cv.IMREAD_GRAYSCALE)
            obj_image_current=obj_image_current[self.object_indeces[i]]
            radius=round(np.floor(self.object_trajectory[i,-1]))
            M=self.confectionery.sell_cake(radius,obj_image_current.shape[0],obj_image_current.shape[0])
            obj_image_current[M==0]=0
            pad_length=self.rmax-(obj_image_current.shape[0]-1)//2
            obj_image_current=np.pad(obj_image_current, [(pad_length,pad_length),(pad_length,pad_length)])
            self.object_tensor[:,:,i]=obj_image_current

    def load_derivatives_initial(self, frame_portion=1.0):
        """
        Load data_derivatives so we can cluster them

        Parameters
        ----------
        frame_portion : float between (0,1)
            portion of frames that will be used, standard value is 1.0 (all frames of object)
        Returns
        ----------
        current_index : int
            current index of frame
        """
        N=round(len(self.image_stream)*frame_portion)
        for i in range(N):
            obj_image_current=cv.imread(self.image_stream[i], cv.IMREAD_GRAYSCALE)
            obj_image_current=obj_image_current[self.object_indeces[i]]
            obj_derivative=self.lowpass.derivative_filter(obj_image_current)

            #obj_image_current=self.lowpass.current_filter(obj_image_current)
            #obj_derivative=RadialDerivative(obj_image_current, self.confectionery)
            #obj_derivative=obj_derivative.calculate_derivative(valid=True)

            #obj_derivative=obj_derivative.calculate_derivative_magnitude(valid=True)
            #obj_derivative=obj_derivative.calculate_derivative_squared(valid=True)
            self.derivatives[i]=obj_derivative
            if self.rmax<(obj_derivative.shape[0]-1)//2:
                self.rmax=(obj_derivative.shape[0]-1)//2
        self.current_index_derivative=i
        return self.current_index_derivative
    
    def load_next_frame_into_derivatives(self):
        """
        Load next frame into derivative so we can cluster them

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
        obj_image_current=obj_image_current[self.object_indeces[self.current_index]]

        obj_derivative=self.lowpass.derivative_filter(obj_image_current)
        
        #obj_image_current=self.lowpass.current_filter(obj_image_current)
        #obj_derivative=RadialDerivative(obj_image_current, self.confectionery)
        #obj_derivative=obj_derivative.calculate_derivative(valid=True)

        #obj_derivative=obj_derivative.calculate_derivative_magnitude(valid=True)
        #obj_derivative=obj_derivative.calculate_derivative_squared(valid=True)
        self.derivatives[self.current_index]=obj_derivative
        if self.rmax<obj_derivative.shape[0]:
            self.rmax=obj_derivative.shape[0]
        return self.current_index

    def load_derivative_on_radius(self):
        self.data_derivative_r=dict()
        for i in range(len(self.derivatives)):
            current_derivative=self.derivatives[i]
            for j in range((current_derivative.shape[0]+1)//2):
                if j==0:
                    index=self.confectionery.sell_cake(j,current_derivative.shape[0],current_derivative.shape[0])
                else:
                    index=self.confectionery.sell_donut(j,current_derivative.shape[0],current_derivative.shape[0])
                pixel_values=current_derivative[index==1]
                if len(self.data_derivative_r)<j+1:
                    self.data_derivative_r[j]=pixel_values
                else:
                    self.data_derivative_r[j]=np.append(self.data_derivative_r[j],pixel_values)
        return self.data_derivative_r
    
    def hist_derivative_on_radius(self):
        for i in range(1,len(data_derivative_r)):
            ax=plt.figure()
            plt.title("radius: "+str(i))
            data=data_derivative_r[i]
            gauss=GaussianMixture(n_components=1, random_state=0).fit(data.reshape((-1,1)))
            gauss_mean = gauss.means_.reshape((-1))
            gauss_variance = gauss.covariances_.reshape((-1))
            Number_bins=300
            mean=np.mean(data)
            variance=np.sum((data-mean)**2)/(data.size-1)
            min_edge = round(np.floor(mean-2*np.sqrt(variance)))
            max_edge = round(np.ceil(mean+2*np.sqrt(variance)))
            bins=np.linspace(min_edge, max_edge, Number_bins)
            n, bins, patches = plt.hist(data, bins=bins, color = "blue")
            # plot gaussian
            x = np.linspace(min_edge,max_edge, 1000)
            plt.plot(x, data.shape[0]*norm.pdf(x, gauss_mean, np.sqrt(gauss_variance)))
        plt.show()

    def train_2_means(self, plot_hist=False):
        """
        Train 2 means clustering model

        Parameters
        ----------
        plot_hist : boolean
            bool to decide if we plot or not
        """
        self.kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(self.data.reshape((-1,1)))
        self.labels_kmeans=self.kmeans.labels_
        self.centroids_kmeans=self.kmeans.cluster_centers_.reshape((-1))
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
        if plot_hist:
            plt.figure()
            for k in range(number_of_gaussian):
                n, bins, patches = plt.hist(self.data[self.labels_gmm==k], bins=np.arange(min(self.data), max(self.data) +1, 1), color = self.cmap[k])
                d=np.abs(bins.reshape((-1,1))-self.means_gmm.reshape((1,-1)))
                for j in range(self.means_gmm.shape[0]):
                    i=np.argmin(d[:,j])
                    patches[i].set_fc('r')
                # plot gmm
                x = np.linspace(0,255, 1000)
                plt.plot(x, self.priors_gmm[k]*self.data.shape[0]*norm.pdf(x, self.means_gmm[k], np.sqrt(self.variance[k])))
            plt.show()


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
        if plot_hist:
            plt.figure()
            for k in range(2):
                n, bins, patches = plt.hist(self.data[self.labels_gmm_cutoff_filter==k], bins=np.arange(min(self.data), max(self.data) +1, 1), color = self.cmap[k])
                d=np.abs(bins.reshape((-1,1))-self.means_gmm_cutoff_filter.reshape((1,-1)))
                for j in range(self.means_gmm_cutoff_filter.shape[0]):
                    i=np.argmin(d[:,j])
                    patches[i].set_fc('r')
                # plot gmm
                x = np.linspace(0,255, 1000)
                plt.plot(x, self.priors_gmm_cutoff_filter[k]*self.data.shape[0]*norm.pdf(x, self.means_gmm_cutoff_filter[k], np.sqrt(self.variance_cutoff_filter[k])))
            plt.show()


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
    lp=LowPass(koppenrath)
    # cluster
    half_shelf_cluster=Half_shelf_cluster(object_trajectory, stream_string, confectionery=koppenrath, lowpass=lp)
    half_shelf_cluster.load_derivatives_initial(frame_portion=0.3)

    print(len(half_shelf_cluster.derivatives))
    for i in range(len(half_shelf_cluster.derivatives)):
        print(half_shelf_cluster.derivatives[i].shape)
    
    data_derivative_r=half_shelf_cluster.load_derivative_on_radius()


