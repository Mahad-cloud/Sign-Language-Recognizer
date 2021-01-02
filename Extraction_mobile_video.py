import cv2
import numpy as np

videos_path = '/media/mahad/BE3233F53233B0EF/dataset_videos/'
writing_path = '/media/mahad/BE3233F53233B0EF/fyp/mobile_word_based_videos/dataset_word/testing_hai/you/'

cap = cv2.VideoCapture(videos_path + 'you_home_hadi.3gp')
i = 0
file_counter = 1
while True:
    ret, frame = cap.read()

    # print(np.array(frame).shape[1])
    # frame = cv2.resize(frame, (320, 180))
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if i % 17 == 0:
        print(i)
        cv2.imwrite(
            writing_path + 'image_' + 'testing' + str(
                file_counter) + '.png', frame)
        file_counter += 1
        print('image number: ', file_counter)
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(0) == 'q':
    #     cv2.destroyAllWindows()
    i += 1

    # frame = cv2.resize(frame, ())
