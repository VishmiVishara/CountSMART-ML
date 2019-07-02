from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import utils
import imutils

import requests

from PIL import Image

# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
# weights_path='config/yolov3_34000.weights'
# weights_path='config/yolov3_111000.weights'
# weights_path='config/yolov3_final_1.weights'
class_path='config/coco.names'
# class_path='config/people_detection.names'
img_size=416
# conf_thres=0.0001
conf_thres=0.6
nms_thres=0.00001
# nms_thres=0.5

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])

# BoxiFRGateContour = np.array([[0, 0], [700, 0], [700, 1080], [1600, h], [1600, 0], [1920, 0], [1920, 1080], [0, 1080]])
# BoxiFRGateContour = np.array([[0, 0], [700, 0], [700, 1080], [0, 1080]])
BoxiFRGateContour = np.array([[0, 0], [1920, 0], [1920, 300], [700, 300], [700, 800], [1920, 800], [1920, 1080], [0, 1080]])

def getFrame(frame):
    cv2.fillPoly(frame, pts=[BoxiFRGateContour], color=(255, 255, 255))
    # frame = imutils.resize(frame, width=640)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = np.repeat(frame[..., np.newaxis], 3, -1)
    # frame = cv2.filter2D(frame, -1, kernel)  # applying the sharpening kernel to the input image & displaying it.

    return frame

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections


# videopath = "videos/BOXIFRGATE_June7th2019_3-00-00pm-4-00-00pm_cut.mp4"
# videopath = "videos/BOXIFRGATE_June7th2019_5-00-00pm-6-00-00pm_cut.mp4"
# videopath = "videos/try1_r.mp4"
videopath = "videos/ClearPix Camera Grocery Store Front Door.mp4"
# videopath = 0

import cv2
from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh))

frames = 0
starttime = time.time()

centerx = 0
centery = 0

totalUp = 0
totalDown = 0

# W = 1920
# H = 1080
W = 1280
H = 720

#tech - ranja
OBJ_ARR = []
direction = 0
previous_detection = {}

CENTROIDS_BY_OBJECTID   = []

in_url = 'http://192.168.43.35:8080/ML_Backend_3_0_war_exploded/count/save/in'
out_url = 'http://192.168.43.35:8080/ML_Backend_3_0_war_exploded/count/save/out'
headers = {'content-type': 'text/plain'}

while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    # frame = getFrame(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)[0]
    # print(detect_image(pilimg))

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # frame = getFrame(frame)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

    if detections is not None:

        # tracked_objects = mot_tracker.update(detections.cpu())
        tracked_objects = mot_tracker.update(detections.cpu().numpy())
        # print(tracked_objects)

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)

        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:



            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            # print(x1, x2, y1, y2, str(x1 + box_w), str(y1 + box_h))
            centerx = (x1 + x1 + box_w)//2
            centery = (y1 + y1 + box_h)//2
            center = (centerx, centery)
            cv2.circle(frame, center, 5, color, -1)


            # check to see if the object has been counted or not
            # if not tracked_objects.counted:
            if obj_id not in OBJ_ARR:

                if obj_id in previous_detection.keys():

                    direction = center[1] - previous_detection[obj_id][1]

                    previous_detection.update({obj_id: center})

                    # if the direction is negative (indicating the object is moving up) AND the centroid is above the center line,
                    # count the object
                    if direction < 0 and center[1] < H // 2:
                        totalUp += 1
                        r = requests.get(in_url, headers=headers)
                        # tracked_objects.counted = True
                        OBJ_ARR.append(obj_id)

                    # if the direction is positive (indicating the object is moving down) AND the centroid is below the center line,
                    # count the object
                    elif direction > 0 and center[1] > H // 2:
                        totalDown += 1
                        r = requests.get(out_url, headers=headers)				
                        # tracked_objects.counted = True
                        OBJ_ARR.append(obj_id)

                else:
                    previous_detection.update({obj_id: center})

                if len(previous_detection) > 20:
                    # print(list(previous_detection)[0])
                    previous_detection.pop(list(previous_detection)[0], None)

                # print(previous_detection)

            if len(OBJ_ARR) > 20:
                OBJ_ARR.pop(0)

            # print(OBJ_ARR)



            obj_id = obj_id % 100

            # print(previous_detection)



            # centroids.append([obj_id, center])

            # if obj_id in OBJ_DIREC:
            #     y = [c[1] for c in centroids]
            #     direction = center[1] - np.mean(y)
            #     centroids.append([obj_id, center])
            #
            # else:
            #     OBJ_DIREC.append(obj_id)

            # print(previous_detection)
            # print(direction)


            # check to see if the object has been counted or not
            # if not tracked_objects.counted:
            # if obj_id not in OBJ_ARR:
            #     # if the direction is negative (indicating the object is moving up) AND the centroid is above the center line,
            #     # count the object
            #     if direction < 0 and center[1] < H // 2:
            #         totalUp += 1
            #         # tracked_objects.counted = True
            #         OBJ_ARR.append(obj_id)
            #
            #     # if the direction is positive (indicating the object is moving down) AND the centroid is below the center line,
            #     # count the object
            #     elif direction > 0 and center[1] > H // 2:
            #         totalDown += 1
            #         # tracked_objects.counted = True
            #         OBJ_ARR.append(obj_id)


            cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
            cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H // 2 - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Stream', frame)
    outvideo.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    # ch = 0xFF & cv2.waitKey(1)
    # if ch == 27:
    #     break

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame", totaltime, "total_time")
cv2.destroyAllWindows()
outvideo.release()
