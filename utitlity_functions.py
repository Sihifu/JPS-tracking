import numpy as np
from scipy.spatial.transform import Rotation as R

def caluclate_normal_vector_from_contour(edge_pixels_local_3d):
    """
    Calculate Normal vector given edge_pixels_local_3d
    
    Parameters
    ----------
    edge_pixels_local_3d: np.array (N,3)

    Returns
    ----------
    normal vector: np.array (3,)
    """
    svd = np.linalg.svd(edge_pixels_local_3d)
    n=svd[2][-1,:]
    return n

def calculate_rotations_from_normal_vector_trajectory(normal_vector_trajectory):
    """
    Calculate rotation matrices
    
    Parameters
    ----------
    normal_vector_trajectory: np.array (3,N)

    Returns
    ----------
    rotation_trajectory: dictionary of length N-1 with rotation object
    """
    rotation_trajectory=dict()
    for i in range(normal_vector_trajectory.shape[1]-1):
        n1=normal_vector_trajectory[:,i]
        n2=normal_vector_trajectory[:,i+1]
        n_axis=np.cross(n1,n2)
        if np.linalg.norm(n_axis)<1e-6:
            # almost no rotation
            # rot=R.from_rotvec(np.zeros((3,)))
            if np.linalg.norm(n1-n2)<1e-6:
                # almost parallel normal vector
                rot=R.from_rotvec(np.zeros((3,)))
            else:
                # almost antiparallel parallel normal vector
                A=np.concatenate((n1,n2),axis=0).reshape((2,-1))
                U,_,_=np.linalg.svd(A.T)
                n_axis=U[:,1].reshape(-1)
                rot=R.from_rotvec(np.pi*n_axis)
        else:
            rotation_angle=np.arctan2(np.linalg.norm(n_axis),np.dot(n1,n2))
            n_axis=n_axis/np.linalg.norm(n_axis)
            axis_rot=rotation_angle*n_axis
            rot=R.from_rotvec(axis_rot)
        rotation_trajectory[i]=rot
    return rotation_trajectory

def calculate_global_rotations_from_normal_vector_trajectory(normal_vector_trajectory, n_0):
    """
    Calculate global rotation matrices
    
    Parameters
    ----------
    normal_vector_trajectory: np.array (3,N)
    n_0: np.array (3,1) starting configuration

    Returns
    ----------
    gloabl rotation_trajectory: dictionary of length N with rotation object
    """
    global_rotation_trajectory=dict()
    for i in range(normal_vector_trajectory.shape[1]):
        n=normal_vector_trajectory[:,i]
        n_axis=np.cross(n_0.reshape(-1),n)
        if np.linalg.norm(n_axis)<1e-6:
            if np.linalg.norm(n_0.reshape(-1)-n)<1e-6:
                # almost parallel normal vector
                rot=R.from_rotvec(np.zeros((3,)))
            else:
                # almost antiparallel parallel normal vector
                n_axis=np.copy(n_0).reshape(-1)
                j=np.argsort(np.abs(n_axis))
                n_axis[j==0]=0
                n_axis[j==2]=-n_axis[j==2]
                n_axis[j[1:]]=n_axis[j[:0:-1]]
                n_axis=n_axis/np.linalg.norm(n_axis)
                rot=R.from_rotvec(np.pi*n_axis)
        else:
            rotation_angle=np.arctan2(np.linalg.norm(n_axis),np.dot(n_0.reshape(-1),n))
            n_axis=n_axis/np.linalg.norm(n_axis)
            axis_rot=rotation_angle*n_axis
            rot=R.from_rotvec(axis_rot)
        global_rotation_trajectory[i]=rot
    return global_rotation_trajectory


def calculate_global_rotations_from_local_rotation_trajectory(rotation_trajectory, r_0):
    """
    Calculate global rotation matrices
    
    Parameters
    ----------
    rotation_trajectory: dictionary of length N-1 with local rotation object (from one frame to next frame)
    r_0: rotation object which is frist roation configuration

    Returns
    ---------
    global_rotation_trajectory: dictionary of length N with rotation object (from first frame to frame n)
    """
    global_rotation_trajectory=dict()
    global_rotation_trajectory[0]=r_0
    for i in range(1,len(rotation_trajectory)+1):
        r=rotation_trajectory[i-1].as_matrix()@global_rotation_trajectory[i-1].as_matrix()
        global_rotation_trajectory[i]=R.from_matrix(r)
    return global_rotation_trajectory

def smooth_edge_pixels_local_trajectory(edge_pixels_local_trajectory):
    for i in range(len(edge_pixels_local_trajectory)):
        if edge_pixels_local_trajectory[i].shape[0]<=5:
            edge_pixels_local_trajectory[i]=np.empty((0,3))
    return edge_pixels_local_trajectory

def non_max_suppression(I_magn, I_theta):
    """
    Applies Non maximun suppresion and returns the resulting Matrix
     
    Parameters
    ----------
    I_magn: np.array (M,N,dtype=np.float32)
        Array with derivative magnitudes
    I_theta: np.array (M,N,dtype=np.float32)
        Array with derivative angles in radiants

    Returns
    ----------
    output: np.array (M-1,N-1,dtype=np.float32)
        Array with thinned out edges
    """

    phi=np.copy(I_theta)
    phi[phi<0]=phi[phi<0]+np.pi

    # number represent the following directions (and vice versa)
    # 0 == West to East
    # 1 == Soutwest to Northeast
    # 2 == North to South
    # 3 == Northwest to Southeast
    
    Directions=np.zeros(phi.shape,dtype=int)
    for i in range(3):
        lower_bound=(np.pi/8)+i*(np.pi/4)
        upper_bound=(np.pi/8)+(i+1)*(np.pi/4)
        index=(phi >= lower_bound) & (phi <= upper_bound) 
        Directions[index]=i+1
    out=np.zeros_like(I_magn)
    for y in range(1,phi.shape[0]-1):
        for x in range(1,phi.shape[1]-1):
            b=I_magn[y,x]
            if Directions[y,x]==0:
                a=np.array(I_magn[y,x-1:x+2])
                if np.max(a)==b:
                    out[y,x]=b
            elif Directions[y,x]==1:
                a=np.array([I_magn[y-1,x-1],b,I_magn[y+1,x+1]])
                if np.max(a)==b:
                    out[y,x]=b
            elif Directions[y,x]==2:
                a=np.array(I_magn[y-1:y+2,x])
                if np.max(a)==b:
                    out[y,x]=b
            elif Directions[y,x]==3:
                a=np.array([I_magn[y+1,x-1],b,I_magn[y-1,x+1]])
                if np.max(a)==b:
                    out[y,x]=b
    out[1:-1,1:-1]
    return out


def hysteresis_tracking(pixels_strength):
    """
    Returns the output of Canny Edge Detector

    Parameters
    ----------
    pixels_strength: np.array (M,N) with int values 0,1 or 2
        Array with pixel strenght values, 2 is strong, 1 is weak, 0 is non edge

    Returns
    ----------
    output: np.array (M,N,dtype=np.uint8) with values 255 or 0
        Array with canny edges
    """
    pixels_weak_coordinates=np.argwhere(pixels_strength==1).astype(int)
    pixels_weak_coordinates[:,[0, 1]]=pixels_weak_coordinates[:,[1, 0]]
    pixels_strong_coordinates=np.argwhere(pixels_strength==2).astype(int)
    pixels_strong_coordinates[:,[0, 1]]=pixels_strong_coordinates[:,[1, 0]]
    final_edges_coordinates=np.empty((0,2)).astype(int)
    while(pixels_strong_coordinates.size!=0):
        # iterate through strong pixels
        p=pixels_strong_coordinates[0,:]
        neigbour_matrix=np.abs(pixels_weak_coordinates-p.reshape((1,2)))
        neigbour_matrix=neigbour_matrix<=np.ones((1,2))
        neigbour_indeces=neigbour_matrix[:,0] & neigbour_matrix[:,1]
        # pop and push accordingly
        pixels_strong_coordinates=np.concatenate((pixels_strong_coordinates,pixels_weak_coordinates[neigbour_indeces,:]),axis=0)
        pixels_weak_coordinates=pixels_weak_coordinates[neigbour_indeces==0,:]
        final_edges_coordinates=np.concatenate((final_edges_coordinates,p.reshape((1,2))),axis=0)

        if pixels_strong_coordinates.size!=2:
            pixels_strong_coordinates=pixels_strong_coordinates[1:,:]
        else:
            break
    output=np.zeros(pixels_strength.shape,dtype=np.uint8)
    output[final_edges_coordinates[:,1],final_edges_coordinates[:,0]]=255
    return output

def normal_vector_degenerate(clustered_image):
    """
    orients the normal vector of plane in direction of sputtered pixels when no contour could be detected

    Parameters
    ----------
    clustered_image: np.array(M,M, dtype=int) with values (0,1,2) where 1 is sputtered and 2 is unsputtered
        clustered image matrix
    Returns
    ----------
    output: np.array (3,dtype=np.float32)
        normalized oriented normal vector of plane
    """
    if np.sum(clustered_image==1)>np.sum(clustered_image==2):
        return np.array([0,0,-1])
    else:
        return np.array([0,0,1])
    
def orient_normal_plane(n, clustered_image, obj_0, frame_number):
    """
    orients the normal vector of plane in direction of sputtered pixels

    Parameters
    ----------
    n: np.array (3,dtype=np.float32)
        normalized normal vector of plane
    clustered_image: np.array(M,M, dtype=int) with values (0,1,2) where 1 is sputtered and 2 is unsputtered
        clustered image matrix
    obj_0: np.array(N,4, dtype=np.float) 
        tracked object trajectory
    frame_number: int
        corresponding frame number

    Returns
    ----------
    output: np.array (3,dtype=np.float32)
        normalized oriented normal vector of plane
    """
    x=obj_0[frame_number,1]
    y=obj_0[frame_number,2]
    r=obj_0[frame_number,3]
    x_obj_im=round(x-round(r))
    y_obj_im=round(y-round(r))
    x=x-x_obj_im
    y=y-y_obj_im
    edge_pixels_sputtered=np.argwhere(clustered_image==1)[:,[1,0]]
    edge_pixels_unsputtered=np.argwhere(clustered_image==2)[:,[1,0]]
    if edge_pixels_sputtered.shape[0]>=edge_pixels_unsputtered.shape[0]:
        pixels=edge_pixels_sputtered[:,:2]-np.array([x,y]).reshape((1,2))
        z=-np.sqrt(np.abs(r**2-np.sum(pixels**2,axis=1,keepdims=True)))
        pixels=np.concatenate((pixels,z),axis=1)
        if np.sum(np.sum(pixels*n.reshape((1,3)),axis=1)>=0) <= np.sum(np.sum(pixels*-n.reshape((1,3)),axis=1)>0):
            n=-n
    else:
        pixels=edge_pixels_unsputtered[:,:2]-np.array([x,y]).reshape((1,2))
        z=-np.sqrt(np.abs(r**2-np.sum(pixels**2,axis=1,keepdims=True)))
        pixels=np.concatenate((pixels,z),axis=1)
        if np.sum(np.sum(pixels*n.reshape((1,3)),axis=1)>=0) >= np.sum(np.sum(pixels*-n.reshape((1,3)),axis=1)>0):
            n=-n
    return n

def apply_rotation(p_0, rotation_trajectory):
    """
    apply rotation sequence to vector p_0

    Parameters
    ----------
    p_0: np.array (3,dtype=np.float32)
        3d vector
    rotation_trajectory: dictionary of length N-1 with local rotation object (from one frame to next frame)

    Returns
    ----------
    output: np.array (N,3,dtype=np.float32)
        trajectory of points startin g at p_0
    """
    output=np.zeros((len(rotation_trajectory)+1,3))
    output[0,:]=np.copy(p_0)
    for i in range(len(rotation_trajectory)):
        output[i+1,:]=rotation_trajectory[i].apply(output[i,:])
    return output

def distance_rot_on_sphere(p_0,rotation_trajectory,radius):
    """
    starting at vector p_0 on sphere and given a rotation_trajectory
    calculate the travelled distance along the trajectory

    Parameters
    ----------
    p_0: np.array (3,dtype=np.float32)
        3d vector with norm 1
    rotation_trajectory: dictionary of length N-1 with local rotation object (from one frame to next frame)
    radius: float
        radius of sphere
    Returns
    ----------
    output: np.array (N,dtype=np.float32)
        travelled distance, total distance is the last value of output
    """
    p_traj=apply_rotation(p_0, rotation_trajectory)
    axis_rot=np.array([x.as_rotvec() for x in rotation_trajectory.values()])
    output=np.zeros(p_traj.shape[0])
    for i in range(1,output.size):
        p=p_traj[i-1,:]
        v=axis_rot[i-1,:]
        alpha=np.linalg.norm(v)
        if alpha<=1e-6:
            v=p
        else:
            v=v/np.linalg.norm(v)
        r=np.linalg.norm(p-np.dot(p,v)*v)
        output[i]=r*alpha
    output=radius*np.add.accumulate(output)
    return output

def shortest_angles(alpha_1,alpha_2):
    """
    Calculates the shortesr angle between alpha1 and alpha2

    Parameters
    ----------
    alpha_1: np.array (N,dtype=np.float32) in [0,2*np.pi)
    alpha_2: np.array (N,dtype=np.float32) in [0,2*np.pi)    
    Returns
    ----------
    output: np.array (N,dtype=np.float32) in [0,np.pi)
        shorter absolute angle between alpha_1 and alpha_2
    """


def draw_plane_hard_constraint(ax, n):
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    zlim=ax.get_zlim()
    axlim=np.array([xlim,ylim,zlim])
    permutation=np.argsort(np.abs(n))
    n_permuted=n[permutation]
    axlim_permuted=axlim[permutation]
    X1,X2 =np.meshgrid(np.arange(axlim_permuted[0,0],axlim_permuted[0,1]+1),
                np.arange(axlim_permuted[1,0],axlim_permuted[1,1]+1))
    X3=-(n_permuted[0]*X1+n_permuted[1]*X2)/n_permuted[-1]
    X=np.concatenate((X1[:,:,np.newaxis],X2[:,:,np.newaxis]),axis=2)
    X=np.concatenate((X,X3[:,:,np.newaxis]),axis=2)
    X=X[:,:,np.argsort(permutation)]
    return X[:,:,0],X[:,:,1],X[:,:,2]

