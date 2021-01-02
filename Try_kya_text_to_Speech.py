import cv2
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('inception_words.h5')

img_rows = 320
img_cols = 180

class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']

word_labels = ['are', 'me', 'mute', 'need', 'not', 'we', 'where', 'will', 'you']
#
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('output_mobilenet.avi', fourcc, 5, (180, 320))
cap = cv2.VideoCapture('/media/mahad/FYP/mobile_word_based_videos/testing_video.3gp')
# cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

counter = 0
while cap.isOpened():

    ret, frame = cap.read()
    if ret:
        # cv2.rectangle(frame, (100, 100), (280, 420), (255, 255, 255), 2)
        # roi = frame[100:500, 100:500]
        #
        # for image_name in os.listdir('/media/mahad/FYP/mobile_word_based_videos/dataset_word/realtime_test/mixed/'):
        #     frame = cv2.imread('/media/mahad/FYP/mobile_word_based_videos/dataset_word/realtime_test/mixed/' + image_name)
        # frame = cv2.imread('/home/mahad/Desktop/try_you.png')
        # if counter % 3 == 0:
            # print(image_name)
        frame = cv2.resize(frame, (320, 180))
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # img = cv2.resize(frame, (img_cols, img_rows))
        img = image.img_to_array(frame)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255
        pred = np.argmax(model.predict(img))
        print(counter, 'predict returns: ', model.predict(img))
        color = (0, 0, 255)
        # if counter % 5 == 0:
        cv2.putText(frame, word_labels[pred], (50, 50), font, 1.0, color, 2)
        video.write(frame)
        # myobj = gTTS(text=class_labels[pred], lang='en', slow=False)
        # myobj.save("welcome.mp3")
        # mixer.init()
        # mixer.music.load('/home/mahad/PycharmProjects/alphabets/welcome.mp3')
        # mixer.music.play()
        cv2.imshow('Video', frame)
        # cv2.waitKey(0)
        if cv2.waitKey(1) == 'q':
            break
        counter += 1
    else:
        break
print('loop is ended')
cap.release()
video.release()
cv2.destroyAllWindows()
