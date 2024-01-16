import numpy as np 
import cv2 as cv
from segmentation import Segmentation as Segmentation
from background_subtraction_gmm import Backsub as Backsub
from matplotlib import pyplot as plt
import numpy.fft as nf


def main():
    #src=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t011.jpg"), cv.IMREAD_COLOR)
    #src=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t391.jpg"), cv.IMREAD_COLOR)
    src=cv.imread(cv.samples.findFile("Image_Data/Aufnahmen_JPEG_06_07_take_2/Hoang_07_06_2023_Aufnahmen_Rollen_Partikel_Rollen_12_mu_200_002_t410.jpg"), cv.IMREAD_COLOR)
    backsub=Backsub("Video_Data/take_02.avi", history=100)
    backsub.begin_train(max_frames=500)
    segm=Segmentation(backsub)
    segm.segment_image(src)
    object_pixels=segm.get_object_list()
    gray=cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    vector_colors=np.zeros((3,3),np.uint8)
    vector_colors[0,:]=np.array([255,0,0])
    vector_colors[1,:]=np.array([0,128,0])
    vector_colors[2,:]=np.array([0,0,255])
    # clusters
    K=2
    attempts=10
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # object number
    j=0
    object_values= np.float32(gray[object_pixels[j][0][:],object_pixels[j][1][:]].reshape(-1))
    ret,label,center=cv.kmeans(object_values, K, None, criteria, attempts, cv.KMEANS_RANDOM_CENTERS)
    image_clustered=np.zeros(src.shape,np.uint8)
    for k in range(K):
        image_clustered[object_pixels[j][0,(label==k).reshape(-1)],object_pixels[j][1,(label==k).reshape(-1)]]=vector_colors[k,:].reshape((1,1,3))

    x=round(segm.circles[j,0])
    y=round(segm.circles[j,1])
    r=round(segm.circles[j,2])
    m=40
    cut_src=src[y-r-m:y+r+m+1,x-r-m:x+r+m+1,:]
    cut_image_clustered=image_clustered[y-r:y+r+1,x-r:x+r+1,:]
    plt.figure()
    plt.subplot(131),plt.imshow(src)
    plt.title("Original Image")
    plt.subplot(132),plt.imshow(cut_src)
    plt.title('Original Image enlarge'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow( cut_image_clustered)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    Z,Z_alternate=scalarfield_visualize(object_pixels[0],gray,segm.circles[0])
    object_values= np.float32(Z[object_pixels[j][0][:]-y+r,object_pixels[j][1][:]-x+r].reshape(-1))
    ret,label,center=cv.kmeans(object_values, K, None, criteria, attempts, cv.KMEANS_RANDOM_CENTERS)
    image_clustered=np.zeros(src.shape,np.uint8)
    for k in range(K):
        image_clustered[object_pixels[j][0,(label==k).reshape(-1)],object_pixels[j][1,(label==k).reshape(-1)]]=vector_colors[k,:].reshape((1,1,3))

    x=round(segm.circles[j,0])
    y=round(segm.circles[j,1])
    r=round(segm.circles[j,2])
    cut_src=src[y-r-m:y+r+m+1,x-r-m:x+r+m+1,:]
    cut_image_clustered=image_clustered[y-r:y+r+1,x-r:x+r+1,:]
    plt.figure()
    plt.subplot(131),plt.imshow(src)
    plt.title("Original Image")
    plt.subplot(132),plt.imshow(cut_src)
    plt.title('Original Image enlarge'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow( cut_image_clustered)
    plt.title('clustered Image with lowpass'), plt.xticks([]), plt.yticks([])
    plt.show()

def scalarfield_visualize(object_pixels,gray,circle):
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    x_center=round(circle[0])
    y_center=round(circle[1])
    radius=round(circle[2])
    x=np.arange(2*radius+1)+x_center
    y=np.arange(2*radius+1)+y_center
    y=y[::-1]
    X, Y = np.meshgrid(x, y)
    Z=np.zeros(gray.shape,np.uint8)
    Z[object_pixels[0,:],object_pixels[1,:]]=gray[object_pixels[0,:],object_pixels[1,:]]
    Z=Z[y_center-radius:y_center+radius+1,x_center-radius:x_center+radius+1]

    fig=plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.plot_surface(X, Y, Z,cmap=cm.magma,linewidth=0,antialiased=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.title("Heatmap without lowpass filter")

    Z1 = cv.GaussianBlur(Z, (3,3), 1) 
    ax = fig.add_subplot(1, 3, 2, projection='3d')  
    ax.plot_surface(X, Y, Z1,cmap=cm.magma,linewidth=0,antialiased=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.title("Heatmap with gaussian kernel lowpass filter")

    Z2 = lowpass_fourier(Z,5)
    ax = fig.add_subplot(1, 3, 3, projection='3d')  
    ax.plot_surface(X, Y, Z2,cmap=cm.magma,linewidth=0,antialiased=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.title("Heatmap with lowpass filter fourier FFT")
    plt.show()
    return Z1,Z2

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def lowpass_fourier(Z, cut_off):

    M=np.zeros(Z.shape,dtype=bool)
    center = np.array(M.shape)/2.0
    for iy in range(Z.shape[0]):
        for ix in range(Z.shape[1]):
            M[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < cut_off **2
    tf = nf.fftshift(nf.fft2(Z))
    tf_filtered=np.zeros(tf.shape,dtype=complex)
    tf_filtered[M]=tf[M]
    return np.abs(nf.ifft2(nf.ifftshift(tf_filtered))).astype(np.uint8)



if __name__ == "__main__":
    main()