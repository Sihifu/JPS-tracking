import numpy as np
from sklearn.metrics import pairwise_distances as pairwise_distances
from scipy.optimize import linear_sum_assignment as linear_sum_assignment

def match(circles_stream, eps1=50, frame_numb_offset=0):
    """
    Input
    -----
    circles_stream: np.array (#frames,3)
    eps1: match threshold;legitimate match is only valid if objects are less than eps1 pixels away
    frame_numb_offset:
    Output
    ------
    matched_objects: Dictionary with stream of particle, values are np array of (frames,4); frame_number,xpos,ypos,radius
    """
    # hypothesis next image circles will match with smallest distance, since frame rate is high
    # a legitimate match is only valid if objects are less than eps1 pixels away
    frame_numb_offset=0
    while True:
        if circles_stream[frame_numb_offset] is not None:
            break
        frame_numb_offset+=1
    circles_current=circles_stream[frame_numb_offset][:,:-1]
    # initialize matched objects
    a=np.concatenate((frame_numb_offset*np.ones(circles_current.shape[0]).reshape((-1,1)),circles_stream[frame_numb_offset]),axis=1)
    matched_objects=dict(enumerate(a[:,np.newaxis,:]))
    # initialize objects_ids to keep track
    objects_ids_current=np.arange(circles_current.shape[0])
    total_num_objects=circles_current.shape[0]
    for i in range(frame_numb_offset,len(circles_stream)-1):
        frame_numb_offset+=1
        if circles_stream[i] is not None:
            circles_current=circles_stream[i][:,:-1]
        if circles_stream[frame_numb_offset] is not None:
            circles_next=circles_stream[frame_numb_offset][:,:-1]
        else:
            circles_current=None
            objects_ids_current=np.empty(0,dtype=int)
            continue
        if circles_stream[i] is not None:
            D = pairwise_distances(circles_next,circles_current)
            # extend bipartite graph to match new ids if too big
            b=np.full((circles_next.shape[0],circles_next.shape[0]),np.inf)
            np.fill_diagonal(b, eps1)
            D = np.concatenate((D,b),axis=1)
            # extend objects_ids_current
            objects_ids_current=np.append(objects_ids_current,np.arange(total_num_objects,total_num_objects+circles_next.shape[0]))
            _, col_ind = linear_sum_assignment(D)
            objects_ids_next=[]
            for index_next,optimal_index in enumerate(col_ind):
                a=np.concatenate((np.array([frame_numb_offset]),circles_stream[frame_numb_offset][index_next,:])).reshape((1,-1))
                id=objects_ids_current[optimal_index]
                if id>=total_num_objects:
                    # create new id
                    matched_objects[total_num_objects]=a
                    objects_ids_next.append(total_num_objects)
                    total_num_objects+=1
                else:
                    matched_objects[id]=np.vstack((matched_objects[id],a))
                    objects_ids_next.append(id)
            objects_ids_current=np.array(objects_ids_next)
        else:
            objects_ids_next=[]
            for circle in circles_stream[frame_numb_offset]:
                a=np.concatenate((np.array([frame_numb_offset]),circle)).reshape((1,-1))
                matched_objects[total_num_objects]=a
                objects_ids_next.append(total_num_objects)
                total_num_objects+=1
            objects_ids_current=np.array(objects_ids_next)
    return matched_objects

def match_Jaq(circles_stream, eps1=50, frame_numb_offset=0):
    """
    Input
    -----
    circles_stream: np.array (#frames,3)
    eps1: match threshold;legitimate match is only valid if objects are less than eps1 pixels away
    frame_numb_offset:
    Output
    ------
    matched_objects: Dictionary with stream of particle, values are np array of (frames,4); frame_number,xpos,ypos,radius
    """
    # hypothesis next image circles will match with smallest distance, since frame rate is high
    # a legitimate match is only valid if objects are less than eps1 pixels away
    frame_numb_offset=0
    while True:
        if circles_stream[frame_numb_offset] is not None:
            break
        frame_numb_offset+=1
    circles_current=circles_stream[frame_numb_offset][:,:-1]
    # initialize matched objects
    a=np.concatenate((frame_numb_offset*np.ones(circles_current.shape[0]).reshape((-1,1)),circles_stream[frame_numb_offset]),axis=1)
    matched_objects=dict(enumerate(a[:,np.newaxis,:]))
    # initialize objects_ids to keep track
    objects_ids_current=np.arange(circles_current.shape[0])
    total_num_objects=circles_current.shape[0]
    for i in range(frame_numb_offset,len(circles_stream)-1):
        frame_numb_offset+=1
        if circles_stream[i] is not None:
            circles_current=circles_stream[i][:,:]
        if circles_stream[frame_numb_offset] is not None:
            circles_next=circles_stream[frame_numb_offset][:,:]
        else:
            circles_current=None
            objects_ids_current=np.empty(0,dtype=int)
            continue
        if circles_stream[i] is not None:
            D_11 = pairwise_distances(circles_current,circles_next)
            # extend bipartite graph to match new ids if too big
            D_12=np.full((D_11.shape[0],D_11.shape[0]),np.inf)
            np.fill_diagonal(D_12, eps1)
            D_21=np.full((D_11.shape[1],D_11.shape[1]),np.inf)
            np.fill_diagonal(D_21, eps1)
            D_upper = np.concatenate((D_11,D_12),axis=1)
            D_lower = np.concatenate((D_21,D_11.T),axis=1)
            D=np.concatenate((D_upper,D_lower),axis=0)

            _, col_ind = linear_sum_assignment(D)
            objects_ids_next=[]
            for index_next,optimal_index in enumerate(col_ind):
                try:
                    if optimal_index>=D_11.shape[1]:
                        # terminate track
                        continue
                    a=np.concatenate((np.array([frame_numb_offset]),circles_stream[frame_numb_offset][optimal_index,:])).reshape((1,-1))
                    if index_next>=D_11.shape[0]:
                        # start new track
                        id=total_num_objects
                        total_num_objects=total_num_objects+1
                        matched_objects[id]=a
                    else:
                        # concatenate track
                        id=objects_ids_current[index_next]
                        matched_objects[id]=np.vstack((matched_objects[id],a))
                    objects_ids_next.append(id)
                except:
                    print("hmm")
            objects_ids_current=np.array(objects_ids_next)
        else:
            objects_ids_next=[]
            for circle in circles_stream[frame_numb_offset]:
                a=np.concatenate((np.array([frame_numb_offset]),circle)).reshape((1,-1))
                matched_objects[total_num_objects]=a
                objects_ids_next.append(total_num_objects)
                total_num_objects+=1
            objects_ids_current=np.array(objects_ids_next)
    return matched_objects


def erase_border_circles(circles_stream, width, height):
    for j,circles in enumerate(circles_stream):
        if type(circles) is np.ndarray:
            left_circles=np.floor(circles[:,:2]-circles[:,-1].reshape((-1,1)))
            right_circles=np.ceil(circles[:,:2]+circles[:,-1].reshape((-1,1)))
            left_corner=np.array([2,2]).reshape((1,2))
            right_corner=np.array([width-2, height-2]).reshape((1,2))
            l=left_circles>=left_corner
            l=l.all(axis=1)
            r=right_circles<=right_corner
            r=r.all(axis=1)
            index=np.logical_and(l, r)
            if index.any():
                circles_stream[j]=circles[index,:]
            else:
                circles_stream[j]=None
    return circles_stream

def erase_short_objects(matched_objects,min_num_frames=20):
    obj_id=0
    new_matched_objects={}
    for i in range(len(matched_objects)):
        if matched_objects[i].shape[0]<=min_num_frames:
            continue
        else:
            new_matched_objects[obj_id]=matched_objects[i]
            obj_id+=1
    return new_matched_objects

if __name__=="__main__":
    circle_stream=[]
    x1=np.array([[1,2,3]])
    x2=np.array([[4,4,3]])
    X=np.vstack((x1,x2))
    y2=np.array([[2,2,-1]])
    y1=np.array([[5,5,1]])
    y3=np.array([[30,30,1]])
    Y=np.vstack((y1,y2,y3))
    circle_stream.append(X)
    circle_stream.append(Y)
    circle_stream.append(X)
    print(match(circle_stream))
    circle_stream=erase_border_circles(circle_stream, 20, 20)
    print(match(circle_stream))
    circle_stream=[np.array([[1226.5, 1543.5,   17.2],
       [ 131.5,  130.5,   17.1]], dtype=np.float32), np.array([[2025.5,  746.5,   17.2],
       [ 127.5,  129.5,   17.6],
       [1221.5, 1547.5,   17.2]], dtype=np.float32)]
    print(match(circle_stream))
    circle_stream=[X,Y,None,None,X]
    print(match(circle_stream))

    