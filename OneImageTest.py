import os
import joblib as jb
import cv2
import numpy as np
from tensorflow.keras.models import load_model

path = '/media/mahad/BE3233F53233B0EF/fyp/mobile_word_based_videos/dataset_word/diverse_dataset/train/traini/are/are_5.png'

def dictionary():
    count = 0
    lookup = dict()
    for j in os.listdir('/media/mahad/BE3233F53233B0EF/fyp/mobile_word_based_videos/dataset_word/diverse_dataset'
                        '/train/testic/'):
        lookup[count] = j
        count += 1
    print(lookup)
    return lookup


def loading_model():
    # read_model = jb.load('model_vgg.h5')
    read_model = load_model('model_VGG16_9_classes_diverse.h5')
    print(read_model.summary())
    return read_model




# def imageProcessing(image):
#     image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     image_hsv = cv2.resize(image_hsv, (480, 270))
#     hue = image_hsv[:, :, 0]
#     satr = image_hsv[:, :, 1]
#     value = image_hsv[:, :, 2]
#     hsv_images = np.concatenate((hue, satr, value), axis=1)
#     _, hue_thresh = cv2.threshold(hue, 10, 255, cv2.THRESH_BINARY_INV)
#     _, satr_thresh = cv2.threshold(satr, 40, 255, cv2.THRESH_BINARY)
#     skin_image = cv2.bitwise_and(hue_thresh, satr_thresh)
#     cv2.imshow('skin_image', skin_image)
#     if cv2.waitKey(0) == 27:
#         cv2.destroyAllWindows()
#     return skin_image


def read_image():
    image = cv2.imread(path)
    image = cv2.resize(image, (180, 320))
    # image = imageProcessing(image)
    image = np.array(image)
    x_data.append(image)
    return image, x_data


def image_reshaping(x_data):
    x_data = np.array(x_data)
    # x_data = x_data.reshape(1, 320, 180, 1)
    return x_data


def Normalization(x_data):
    x_data = x_data/255
    return x_data


def Prediction(x_data, model, lookup):
    predictions = model.predict(x_data)
    print('index of predictive alphabet: ', np.argmax(predictions))
    print(lookup[np.argmax(predictions)])


x_data = []

lookup = dictionary()
model = loading_model()
image = read_image()
x_data = image_reshaping(x_data)
x_data = Normalization(x_data)
Prediction(x_data, model, lookup)




