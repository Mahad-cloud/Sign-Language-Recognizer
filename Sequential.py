import os
import joblib as jb
import numpy as np
from keras import layers
from keras import models
from keras.layers import Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import time


index = 1
model_version = "2"
lookup = dict()
reverselookup = dict()
count = 0
input_dir = '/media/mahad/BE3233F53233B0EF/Live_feed/train/'
epcohes = 10
batch_size = 64
x_data = []
y_data = []
datacount = 0
model = models.Sequential()
testing_data = []
verifying_data = []
label_encoder = LabelEncoder()
uniques = []
ids = []
Predictive_output = []


def dictionaries_for_labels(count):
    for j in os.listdir(input_dir):
        lookup[j] = count
        reverselookup[count] = j
        count += 1
    return lookup, reverselookup


def imageProcessing(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv = cv2.resize(image_hsv, (480, 270))
    hue = image_hsv[:, :, 0]
    satr = image_hsv[:, :, 1]
    value = image_hsv[:, :, 2]
    hsv_images = np.concatenate((hue, satr, value), axis=1)
    _, hue_thresh = cv2.threshold(hue, 10, 255, cv2.THRESH_BINARY_INV)
    _, satr_thresh = cv2.threshold(satr, 40, 255, cv2.THRESH_BINARY)
    skin_image = cv2.bitwise_and(hue_thresh, satr_thresh)
    # cv2.imshow('skin_image', skin_image)
    # if cv2.waitKey(0) == 27:
    #     cv2.destroyAllWindows()
    return skin_image


def Input_Output_variables(datacount):
    for folder in os.listdir(input_dir):
        if not folder.startswith('.'):
            count = 0
            for image_name in os.listdir(input_dir + folder + '/'):
                img = cv2.imread(input_dir + folder + '/' + image_name)
                img = cv2.resize(img, (270, 480))
                # img = imageProcessing(img)
                arr = np.array(img)
                x_data.append(arr)
                count = count + 1
                y_value = lookup[folder]
                y_data.append(y_value)
                if count == 200:
                    break
            datacount = datacount + count
            if datacount == 800:
                break
            print('alphabet: ', folder, "total: ", datacount)
    return datacount


def Normalization():
    for i in range(0, len(x_data)):
        x_data[i] = x_data[i] / 255


def data_Reshaping(x_data, y_data):
    x_data = np.array(x_data)
    x_data = x_data.reshape((datacount, 480, 270, 3))
    y_data = np.array(y_data)
    y_data = y_data.reshape(datacount, 1)
    # uniques, ids = np.unique(y_data, return_inverse=True)
    y_data = to_categorical(y_data)
    print(len(y_data))
    return x_data, y_data


def Cnn_Model_Training(model, x_train, y_train, x_test, y_test):
    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(480, 270, 3)))
    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # for i in range(1, 11):
    #     print('epoche number is: ', i)
    hist = model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=1, validation_data=(x_test, y_test))
    PLotting(hist, range(2))
    model.save('model.h5')
    jb.dump(model, 'architect.pkl')
    number_of_epoches = range(epcohes)
    return model, hist, number_of_epoches


def PLotting(hist, epoches):
    print('*****************************HISTORY**************************')
    print(hist.history)
    train_loss = hist.history['loss']
    train_acc = hist.history['acc']
    val_loss = hist.history['val_loss']
    val_acc = hist.history['val_acc']
    plt.figure(1, figsize=(7, 5))
    plt.plot(epoches, train_loss)
    plt.plot(epoches, val_loss)
    plt.xlabel('number of epoches')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.legend(['train', 'val'])
    plt.show()
    plt.grid(True)
    print(plt.style.available)
    plt.style.use(['classic'])
    print('*****************************END******************************')


def Finding_Accuracy(model, x_test, y_test):
    predictions = []
    predictions = np.array(model.predict(x_test))
    print(predictions)
    for i in range(0, len(predictions)):
        Predictive_output.append(reverselookup[np.argmax(predictions[i])])
    # print('Predictive: ', Predictive_output)
    [loss, acc] = model.evaluate(x_train, y_train, verbose=1)
    print("Accuracy:" + str(acc))
    # print("loss:" + str(loss))
    # print(model.summary())


lookup, reverselookup = dictionaries_for_labels(count)

print(reverselookup)

datacount = Input_Output_variables(datacount)

x_data, y_data = data_Reshaping(x_data, y_data)

Normalization()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
print(y_train)
print(y_test)
# y_test = list(y_test)
tests = []
for i in range(0, len(y_test)):
    tests.append(reverselookup[list(y_test[i]).index(1.)])
#
print(np.array(x_train).shape)
print(np.array(y_train).shape, "  ", np.array(y_train)[0])

model, hist, number_of_epoches = Cnn_Model_Training(model, x_train, y_train, x_test, y_test)
# PLotting(hist, number_of_epoches)
Finding_Accuracy(model, x_test, y_test)
print('ActualOutput:', tests)
print('Predictive: ', Predictive_output)

