import os

import cv2

write_path = '/media/mahad/BE3233F53233B0EF/Live_feed/diverse_dataset_26_480_270/train_120_72/'
read_path = '/media/mahad/BE3233F53233B0EF/Live_feed/diverse_dataset_26_480_270/train/'

folders_array = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z']
images_array = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
#
# for folder in os.listdir(path):
count = 0
image_number = 1
image_count = 1
image_legal = True
i = 0
folders_array = ['G', 'H', 'J', 'P']
images_array = ['g', 'h', 'j', 'p']
# for index, folder in enumerate(folders_array):
#     image_count = 1
#     image_number = 1
for folder in os.listdir(read_path):
    image_count = 0
    for image_name in os.listdir(read_path + folder + '/'):
        img = cv2.imread(read_path + folder + '/' + image_name)
        resized_image = cv2.resize(img, (135, 240))
        cv2.imwrite(write_path + folder + '/' + image_name + '.jpg', resized_image)
        print(folder, image_name)
        image_count += 1
        print(image_count)
