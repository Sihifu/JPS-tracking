# Importing all necessary libraries
import cv2
import os

def main():
    # Read the video from specified path
    cam = cv2.VideoCapture("Video_Data/magnetic_video_particles.mp4")
    
    try:
        # creating a folder named data
        if not os.path.exists('Image_Data'):
            os.makedirs('Image_Data')
    
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    
    # frame
    currentframe = 0
    
    while(True):
        
        # reading from frame
        ret,frame = cam.read()
    
        if ret:
            # if video is still left continue creating images
            name = 'Image_Data/magnetic_test_frame/' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()