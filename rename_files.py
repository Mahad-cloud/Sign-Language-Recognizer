import os

folder_path = '/media/mahad/FYP/mobile_word_based_videos/more_5_class_variation/class/we/'

counter = 1

for image_name in os.listdir(folder_path):
    os.rename(folder_path + image_name, folder_path + 'image_we_door_' + str(counter) + '.png')
    counter += 1
