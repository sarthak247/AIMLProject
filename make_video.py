import cv2
import os

image_folder = 'frames_frn/'
video_name = 'udnie_frn.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images2 = []
for img in images:
    images2.append(int(img[:-4]))
images2.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 30, (width,height))

for counter, image in enumerate(images2):
    video.write(cv2.imread(os.path.join(image_folder, str(image) + '.png')))
    print(counter)

cv2.destroyAllWindows()
video.release()