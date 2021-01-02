import matplotlib.pyplot as plt
from keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from PlotLosses import PlotLosses


def PLotting(hist, epoches):
    print('*****************************HISTORY**************************')
    print(hist.history)

    train_loss = hist.history['loss']
    train_acc = hist.history['accuracy']
    val_loss = hist.history['val_loss']
    val_acc = hist.history['val_accuracy']

    plt.figure(1, figsize=(7, 5))
    plt.plot(epoches, train_loss)
    plt.plot(epoches, val_loss)
    plt.xlabel('number of epoches')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.legend(['train', 'val'])
    plt.show()
    plt.grid(True)

    plt.figure(1, figsize=(7, 5))
    plt.plot(epoches, train_acc)
    plt.plot(epoches, val_acc)
    plt.xlabel('number of epoches')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.legend(['train', 'val'])
    plt.show()
    plt.grid(True)
    print('*****************************END******************************')


img_rows, img_cols = 320, 180

VGG = MobileNet(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

for layer in VGG.layers:
    layer.trainable = False


def addTopModelVGG(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = (Flatten())(top_model)
    top_model = Dense(units=256, activation="relu")(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)

    return top_model


num_classes = 9

FC_Head = addTopModelVGG(VGG, num_classes)

model = Model(inputs=VGG.input, outputs=FC_Head)

print(model.summary())

train_datagen = ImageDataGenerator(validation_split=0.2, rescale=1. / 255)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 32

train_dir = '/media/mahad/FYP/mobile_word_based_videos/dataset_word/diverse_dataset/train'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(
    'VGG_words.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1)

earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    restore_best_weights=True,
    patience=10,
    verbose=1)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=5,
                                            verbose=1,
                                            factor=0.2,
                                            min_lr=0.0001)

callbacks = [earlystop, checkpoint, learning_rate_reduction]

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy']
              )

nb_train_samples = 12800
nb_validation_samples = 2500

epochs = 4
batch_size = 64

# plotting = PlotLosses()

history = model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save('VGG_words_mobile_net.h5')

PLotting(history, range(epochs))
