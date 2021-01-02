from __future__ import print_function
import sys
import cv2
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerType(trackerType):
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are: ')
        for t in trackerTypes:
            print(t)

    return tracker


cap = cv2.VideoCapture("/home/mahad/Desktop/horse_racing.mp4")
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P' , 'G'), 10, (1280, 720))
success, frame = cap.read()
if not success:
    print("Failed to load the video file")
    sys.exit(1)

bboxes = []
colors = []
while True:
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if k == 113:  # q is pressed
        break
    print('Selected bounding boxes {}'.format(bboxes))
print('Selected bounding boxes {}'.format(bboxes))

trackerType = "CSRT"

multiTracker = cv2.MultiTracker_create()

for bbox in bboxes:
    multiTracker.add(createTrackerType(trackerTypes[0]), frame, bbox)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    success, boxes = multiTracker.update(frame)
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        out.write(frame)

    cv2.imshow('MultiTracker', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
