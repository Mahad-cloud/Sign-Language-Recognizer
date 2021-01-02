import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from Classification_Report import plot_classification_report

model = load_model('model_Inception_9_classes_diverse.h5')
# model = Sequential(layers=model.layers)
print(model.summary())

img_rows = 320
img_cols = 180

validation_datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 32

train_dir = '/media/mahad/BE3233F53233B0EF/fyp/mobile_word_based_videos/dataset_word/testing_hai/'

validation_generator = validation_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    shuffle="false",
    class_mode='categorical',
    batch_size=32)

# print(np.array(test_generator))
# nb_samples = len(filenames)
# print(filenames)
testing = []
counter = 0
for folder in os.listdir(
        '/media/mahad/BE3233F53233B0EF/fyp/mobile_word_based_videos/dataset_word/testing_hai/'):
    for image in os.listdir(
            '/media/mahad/BE3233F53233B0EF/fyp/mobile_word_based_videos/dataset_word/testing_hai/' + folder + '/'):
        img = cv2.imread(
            '/media/mahad/BE3233F53233B0EF/fyp/mobile_word_based_videos/dataset_word/testing_hai/' + folder + '/' + image)
        img = cv2.resize(img, (180, 320))
        img = np.array(img)
        img = img.reshape(1, 320, 180, 3)
        # img = img.flatten()
        img = img / 255
        predict = model.predict(img)
        print(counter, "    ", np.argmax(predict, axis=1))
        testing.append(np.argmax(predict))
        counter += 1

predictions = np.argmax(predict, axis=1)
confusion = confusion_matrix(validation_generator.classes, testing)
print(confusion)
for i in range(0, len(confusion)):
    print(i, ": ", confusion[i])
# class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
# 'U', 'V', 'W', 'X', 'Y', 'Z']
class_labels = ['mute', 'will', 'not', 'where', 'are', 'give', 'need', 'we', 'you']
cr = classification_report(validation_generator.classes, testing, target_names=class_labels)
print('*******************CLASSIFICATION REPORT**********************')
print(cr)
for i in range(0, len(cr)):
    print(i, ": ", cr[i])
# plot_classification_report(cr)

class_names = []

df_cm = pd.DataFrame(reversed(confusion), index=[i for i in reversed(class_labels)],
                     columns=[i for i in class_labels])
plt.figure(figsize=(15, 15))
sn.heatmap(df_cm, annot=True, square=True, cmap='Greys')
plt.xlabel('Actual Outputs')
plt.ylabel('Predictive outputs')
plt.show()
