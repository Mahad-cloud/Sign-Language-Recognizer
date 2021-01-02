import cv2
import numpy as np

path_video = '/media/mahad/BE3233F53233B0EF/SignLanguage_dataset_videos/videos/fog_wala_videos/VID20200120165507.mp4'
folders_array = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z']
images_array = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']

image_legal = True
i = 0
# for video_name in os.listdir(path_video):
#     print(video_name)
video = cv2.VideoCapture(path_video)
images_count = 1
image_number = 1
image_legal = True

while 1:
    ret, image = video.read()
    print(np.array(image).shape)
    try:
        resize = cv2.resize(image, (270, 480))
    except cv2.error as e:
        image_legal = False
        print('Invalid frame!')
    # resize = cv2.resize(image, (270, 480))
    # frame = cv2.resize(frame, (270, 480))
    frame_rotate_90_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # print(ret)
    # if not ret:
    #     break
    if images_count % 4 == 0 and image_legal == True:
        cv2.imwrite(
            '/media/mahad/New Volume/Courses/Machine learning/Live_feed/train_FOG_wala/' + 'Z' +
            '/' + 'z' + '_FOG_' + str(image_number) + '.jpg', frame_rotate_90_clockwise)
        image_number += 1
    images_count += 1
    print(folders_array[i], ' ', images_count)
    if not image_legal:
        break

i += 1
print('number of images formed are: ', images_count)
video.release()
cv2.destroyAllWindows()

# img = cv2.imread('/media/mahad/New Volume/Courses/Machine learning/Live_feed/F/f_1.jpg')
# print(np.array(img).shape)
# img = cv2.resize(img, (200, 300))
# cv2.imshow('image', img)
# cv2.waitKey(0)
