
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class contour_edge:
    """
    Contour Class; computes the contour from clustered image

    Parameters
    ----------
    clustered_images: np.array(N,M,M)
        with  values (0,1,2), where 0 labels background, 1 labels sputtered pixel, 2 labels unsputtered pixel
    cut_radius: float or int
    confectionery: object of type Confectionery
    """
    def __init__(self,clustered_images, cut_radius, confectionery):
        self.clustered_images=clustered_images
        self.cut_radius=cut_radius
        self.confectionery=confectionery
        self.M_lower=self.confectionery.sell_cake(cut_radius-1, clustered_images.shape[1], clustered_images.shape[2])
        self.M_lower=np.repeat(self.M_lower[np.newaxis,:,:],repeats=clustered_images.shape[0],axis=0)
        self.M=self.confectionery.sell_cake(cut_radius, clustered_images.shape[1], clustered_images.shape[2])
        self.M=np.repeat(self.M[np.newaxis,:,:],repeats=clustered_images.shape[0],axis=0)


    def get_edges(self):
        """
        returns the edges after executing self.computing_edges()

        Returns
        -------
        self.edge_pixels: np.array(M,2,dtype=int)
            pixels that belong to the contour (x,y) coordinates (in image axis) repectively
        """
        self.compute_edges()
        return self.edge_pixels

    def compute_edges(self):
        A=np.pad(self.clustered_images==1,pad_width=[(0,0),(1,1),(1,1)], mode="edge")
        A1=np.logical_xor(np.copy(A[:,1:-1,2:]),A[:,1:-1,1:-1])
        A2=np.logical_xor(np.copy(A[:,1:-1,:-2]),A[:,1:-1,1:-1])
        A3=np.logical_xor(np.copy(A[:,2:,1:-1]),A[:,1:-1,1:-1])
        A4=np.logical_xor(np.copy(A[:,:-2,1:-1]),A[:,1:-1,1:-1])
        B=np.logical_or(A1,A2)
        B=np.logical_or(B,A3)
        B=np.logical_or(B,A4).astype(int)
        B[self.M_lower==0]=0
        self.edge_matrix=np.copy(B)
        index_1=np.logical_and(self.clustered_images==1,B)
        B[index_1]=1
        index_2=np.logical_and(self.clustered_images==2,B)
        B[index_2]=2
        a1=np.argwhere(B==1)
        a1=np.concatenate((a1,np.zeros((a1.shape[0],1))),axis=1)
        a2=[a1[a1[:,0]==i,1:] for i in range(A.shape[0])]
        b1=np.argwhere(B==2)
        b1=np.concatenate((b1,np.ones((b1.shape[0],1))),axis=1)
        b2=[b1[b1[:,0]==i,1:] for i in range(A.shape[0])]
        self.edge_pixels=[np.append(x[:,[1, 0, 2]],y[:,[1, 0, 2]],axis=0).astype(int) for x,y in zip(a2,b2)]
    

    def reduce_edges(self):
        """
        cant reduce properly if 2 different contour edges are 2
         1-2 pixels apart
        """
        for j,current_edge_pixels in enumerate(self.edge_pixels):
            if current_edge_pixels.size==0:
                continue
            self.pixels=np.copy(current_edge_pixels)
            self.clusters=[]
            while self.pixels.size!=0:
                x=self.pixels[0,0]
                y=self.pixels[0,1]
                self.cluster_k=np.array(self.pixels[0,:]).reshape((1,3))
                self.pixels=np.delete(self.pixels,0,axis=0)
                if self.pixels.size!=0:
                    self.reduce_edges_recursion(x,y)
                self.clusters.append(self.cluster_k)
            max_size=0
            imax=0
            for i in range(len(self.clusters)):
                if max_size<self.clusters[i].shape[0]:
                    imax=i
                    max_size=self.clusters[i].shape[0]
            self.edge_pixels[j]=self.clusters[imax]

    def reduce_edges_recursion(self, x, y):
        while True:
            index=self.search_neigbour(x,y,self.pixels)
            if index==-1:
                # no neighbour found
                break
            else:
                # neighbour found
                self.cluster_k=np.append(self.cluster_k, self.pixels[index,:].reshape((1,3)),axis=0)
                x_new=self.pixels[index,0]
                y_new=self.pixels[index,1]
                self.pixels=np.delete(self.pixels,index,axis=0)
                self.reduce_edges_recursion(x_new, y_new)
        return 0

    def search_neigbour(self,x,y,pixels):
        index_1=np.abs(pixels[:,0]-x)==1
        index_2=np.abs(pixels[:,1]-y)==0
        index_x=np.logical_and(index_1,index_2)
        index_1=np.abs(pixels[:,0]-x)==0
        index_2=np.abs(pixels[:,1]-y)==1
        index_y=np.logical_and(index_1,index_2)
        index=np.logical_or(index_x,index_y)
        index=np.where(index==True)[0]
        if index.size==0:
            # No Neighbour found
            return -1
        else:
            return index[0]
        
def cluster_to_edge(clustered_image, cut_radius, M):
    """
    Get contour_pixels from clustered image

    Parameters
    ----------
    clustered_image : np.array (n,m, np.uint8) with values 0 1 2
        Where 0 corrsepondents to background, 1 to first cluster (sputtered), and 2 to second cluster (unsputtered)
    cut_radius : float
        value that filtered the object
    M : np.array of shape clustered image
        mask for object
    Returns
    ----------
    edge_pixels: np.array (n,2)
        the pixels corresponding to the contour, first column corrspons to x value, second column corrspons to y value 
    """
    center=np.array([clustered_image.shape[1]//2,clustered_image.shape[0]//2]).reshape((1,2))
    ring=np.copy(M)
    x=np.arange(clustered_image.shape[0])
    pixels = np.array(np.meshgrid(x, x)).T.reshape(-1, 2)
    index=pixels-center
    index_x=np.copy(index)
    index_y=np.copy(index)
    index_x[:,0]=np.abs(index[:,0])+1
    index_x=np.sum((index_x)**2,axis=1)<=cut_radius**2
    index_y[:,1]=np.abs(index[:,1])+1
    index_y=np.sum((index_y)**2,axis=1)<=cut_radius**2
    index=np.logical_and(index_x,index_y)
    pixels=pixels[index]
    ring[pixels[:,1],pixels[:,0]]=0

    half_A=np.ones_like(clustered_image)
    half_A[clustered_image==1]=1
    half_A[clustered_image==2]=0 
    half_A[ring==1]=3
    half_B=np.ones_like(clustered_image)
    half_B[clustered_image==2]=1
    half_B[clustered_image==1]=0
    half_B[ring==1]=3

    contours_A, _= cv.findContours(half_A, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours_B, _= cv.findContours(half_B, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    max_size=0
    A=np.empty((0,2))
    for i in range(len(contours_A)):
        pixels=np.squeeze(contours_A[i])
        index=pixels-center
        index_x=np.copy(index)
        index_y=np.copy(index)
        index_x[:,0]=np.abs(index[:,0])+1
        index_x=np.sum((index_x)**2,axis=1)<=cut_radius**2
        index_y[:,1]=np.abs(index[:,1])+1
        index_y=np.sum((index_y)**2,axis=1)<=cut_radius**2
        index=np.logical_and(index_x,index_y)
        pixels=pixels[index,:]
        if max_size<pixels.shape[0]:
            A=pixels
            max_size=pixels.shape[0]

    max_size=0
    B=np.empty((0,2))
    for i in range(len(contours_B)):
        pixels=np.squeeze(contours_B[i])
        index=pixels-center
        index_x=np.copy(index)
        index_y=np.copy(index)
        index_x[:,0]=np.abs(index[:,0])+1
        index_x=np.sum((index_x)**2,axis=1)<=cut_radius**2
        index_y[:,1]=np.abs(index[:,1])+1
        index_y=np.sum((index_y)**2,axis=1)<=cut_radius**2
        index=np.logical_and(index_x,index_y)
        pixels=pixels[index,:]
        if max_size<pixels.shape[0]:
            B=pixels
            max_size=pixels.shape[0]
    edge_pixels=np.vstack((A,B))
    return edge_pixels

def cluster_to_edge_or(clustered_image, cut_radius, M):
    """
    Get contour_pixels from clustered image

    Parameters
    ----------
    clustered_image : np.array (n,m, np.uint8) with values 0 1 2
        Where 0 corrsepondents to background, 1 to first cluster, and 2 to second cluster
    cut_radius : float
        value that filtered the object
    M : np.array of shape clustered image
        mask for object
    Returns
    ----------
    edge_pixels: np.array (n,2)
        the pixels corresponding to the contour, first column corrspons to x value, second column corrspons to y value 
    """
    edge_matrix=np.zeros((clustered_image.shape[0],clustered_image.shape[1],2), dtype=int)
    for i in range(1,3):
        A=np.pad(clustered_image==i,pad_width=[(1,1),(1,1)], mode="edge")
        A1=np.logical_xor(np.copy(A[1:-1,2:]),A[1:-1,1:-1])
        A2=np.logical_xor(np.copy(A[1:-1,:-2]),A[1:-1,1:-1])
        A3=np.logical_xor(np.copy(A[2:,1:-1]),A[1:-1,1:-1])
        A4=np.logical_xor(np.copy(A[:-2,1:-1]),A[1:-1,1:-1])
        edge_matrix[:,:,i-1]=np.logical_or(A1,A2)
        edge_matrix[:,:,i-1]=np.logical_or(edge_matrix[:,:,i-1],A3)
        edge_matrix[:,:,i-1]=np.logical_or(edge_matrix[:,:,i-1],A4).astype(int)
    edge_matrix=np.logical_or(edge_matrix[:,:,0],edge_matrix[:,:,1])

    M_lower=A=np.pad(M,pad_width=[(1,1),(1,1)], mode="constant")
    A1=np.logical_and(np.copy(M_lower[1:-1,2:]),M_lower[1:-1,1:-1])
    A2=np.logical_and(np.copy(M_lower[1:-1,:-2]),M_lower[1:-1,1:-1])
    A3=np.logical_and(np.copy(M_lower[2:,1:-1]),M_lower[1:-1,1:-1])
    A4=np.logical_and(np.copy(M_lower[:-2,1:-1]),M_lower[1:-1,1:-1])
    M_lower=np.logical_and(M,A1)
    M_lower=np.logical_and(M_lower,A2)
    M_lower=np.logical_and(M_lower,A3)
    M_lower=np.logical_and(M_lower,A4)
    """
    x=np.arange(clustered_image.shape[0])
    pixels = np.array(np.meshgrid(x, x)).T.reshape(-1, 2)
    center=np.array([clustered_image.shape[1]//2,clustered_image.shape[0]//2]).reshape((1,2))
    index=pixels-center
    index_x=np.copy(index)
    index_y=np.copy(index)
    index_x[:,0]=np.abs(index[:,0])+1
    index_x=np.sum((index_x)**2,axis=1)<=cut_radius**2
    index_y[:,1]=np.abs(index[:,1])+1
    index_y=np.sum((index_y)**2,axis=1)<=cut_radius**2
    index=np.logical_and(index_x,index_y)
    pixels=pixels[index]
    ring[pixels[:,1],pixels[:,0]]=0
    edge_matrix[M==0]=0
    edge_matrix[ring==1]=0
    """
    edge_matrix[M_lower==0]=0
    edge_pixels=np.argwhere(edge_matrix)
    edge_pixels[:,[0, 1]]=edge_pixels[:,[1, 0]]
    return edge_matrix, edge_pixels


