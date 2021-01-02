import os
import PIL
# from subprocess import Popen, PIPE
#
# folderToCheck = '/media/mahad/BE3233F53233B0EF/hand-detection-tutorial/'
# fileExtension = '.png'
#
#
# def checkImage(fn):
#     proc = Popen(['identify', '-verbose', fn], stdout=PIPE, stderr=PIPE)
#     out, err = proc.communicate()
#     exitcode = proc.returncode
#     return exitcode, out, err
#
#
# for directory, subdirectories, files, in os.walk(folderToCheck):
#     for file in files:
#         if file.endswith(fileExtension):
#             filePath = os.path.join(directory, file)
#             code, output, error = checkImage(filePath)
#             if str(code) != "0" or str(error, "utf-8") != "":
#                 print("ERROR " + filePath)
#             else:
#                 print("OK " + filePath)
#
# print("-------------- DONE --------------");
path = '/media/mahad/BE3233F53233B0EF/hand-detection-tutorial/dataset_preparation/sara/'
for image_name in os.listdir(path):
    print(os.stat(path+image_name).st_size, '     ', image_name)
    size = os.stat(path+image_name).st_size
    if str(size) == '729088':
        print('yes')
        break

