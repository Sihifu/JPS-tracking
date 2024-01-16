import cv2
import os

def main():
    image_folder = 'Image_Data/Naja_2'
    video_name = 'Video_Data/Naja_2.avi'
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images=sorted(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 
    video = cv2.VideoWriter(video_name, 0, 50, (width,height),isColor=True)

    for image in images:
        video.write(cv2.imread(os.path.join( image_folder,image)))

    cv2.destroyAllWindows()
    video.release()

if __name__=="__main__":
    main()