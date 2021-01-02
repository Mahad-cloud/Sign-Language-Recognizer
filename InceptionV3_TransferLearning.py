import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from PlotLosses import PlotLosses


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


plot_losses = PlotLosses()

img_rows = 320
img_cols = 180

model = InceptionV3(include_top=False, weights='imagenet', input_shape=(img_rows, img_cols, 3), pooling='max')

for layer in model.layers:
    layer.trainable = False


def addTopModelVGG(bottom_model, num_classes):
    top_model = bottom_model.output
    # top_model = GlobalAveragePooling2D()(top_model)
    # top_model = Dropout(0.3)(top_model)

    top_model = (Dense(1024, activation='relu'))(top_model)
    top_model = (Dense(512, activation='relu'))(top_model)
    top_model = (Dense(256, activation='relu'))(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)

    return top_model


num_classes = 9

FC_Head = addTopModelVGG(model, num_classes)

model = Model(inputs=model.input, outputs=FC_Head)

print(model.summary())

datagen = ImageDataGenerator(validation_split=0.2, rescale=1. / 255)

batch_size = 32
train_dir = '/media/mahad/FYP/mobile_word_based_videos/dataset_word/diverse_dataset/train/traini'

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(
    'model_vgg.h5',
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

callbacks = [plot_losses]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy']
              )

nb_train_samples = 10000
nb_validation_samples = 6840

epochs = 10
batch_size = 64

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

history = model.fit_generator(
    train_generator,
    # steps_per_epoch=10,
    steps_per_epoch= nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[plot_losses])

# test_data = validation_generator.filenames
# nb_test_samples = len(test_data)
# predictions = model.predict_generator(test_data, nb_test_samples)
# predictions = np.argmax(predictions, axis=1)
# print(confusion_matrix(validation_generator.classes, predictions))
# class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#                 'V', 'W', 'X', 'Y', 'Z']
# print(classification_report(validation_generator.classes, predictions, target_names=class_labels))
# PLotting(history, range(epochs))

model.save('inception_words.h5')
