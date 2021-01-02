import os
import cv2
import numpy

folder_path = '/media/mahad/BE3233F53233B0EF/hand-detection-tutorial/dataset_preparation/sara_10/'
writing_path = '/media/mahad/BE3233F53233B0EF/hand-detection-tutorial/dataset_preparation/sara_final/'

file_counter = 0

for image in os.listdir(folder_path):
    img = cv2.imread(folder_path + image)
    img = cv2.resize(img, (90, 160))
    cv2.imwrite(
        writing_path + str(image), img)
    file_counter += 1
    print(file_counter)