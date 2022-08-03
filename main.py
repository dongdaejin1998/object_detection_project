#Import the required libraries for Object detection infernece
import time
import tkinter.messagebox

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
# setting min confidence threshold
# 마우스 이벤트 콜백함수 정의

MIN_CONF_THRESH=.7
#Loading the exported model from saved_model directory
PATH_TO_SAVED_MODEL =r'D:\Tensorflow/workspace\training_demo\exported-models\my_model\saved_model'
print('Loading model...', end='')
start_time = time.time()
# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
# LOAD LABEL MAP DATA
PATH_TO_LABELS=r'D:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                          use_display_name=True)
#Image file for inference
IMAGE_PATH=r'C:/Users/user/Desktop/car/KakaoTalk_20220126_170656251.jpg'
image=cv2.imread(IMAGE_PATH)
im_height,im_width,c=image.shape
#print(im_width,im_height)
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array of shape (height, width, channels), where channels=3 for RGB to feed into tensorflow graph.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
image_np = load_image_into_numpy_array(IMAGE_PATH)
# Running the infernce on the image specified in the  image path
# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image_np)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
detections['num_detections'] = num_detections
# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
#print(detections['detection_classes'])
image_np_with_detections = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=MIN_CONF_THRESH,
      agnostic_mode=False)
# This is the way I'm getting my coordinates

scores = detections['detection_scores']
boxes = detections['detection_boxes']
max_boxes_to_draw = boxes.shape[0]
for i in range(min(max_boxes_to_draw, boxes.shape[0])):

    if scores is None or scores[i] > MIN_CONF_THRESH:
        # boxes[i] is the box which will be drawn
        class_name = category_index[detections['detection_classes'][i]]['name']
        print("This box is gonna get used", boxes[i], detections['detection_classes'][i])  # box_coordinate,box_label
        (ymin, xmin, ymax, xmax) = (boxes[i][0] * im_height, boxes[i][1] * im_width,
                                      boxes[i][2] * im_height, boxes[i][3] * im_width)
        print(ymin, xmin, ymax, xmax)  # y,x,h,w
#image_np_with_detections=cv2.cvtColor(image_np_with_detections,cv2.COLOR_BGR2RGB)
cv2.rectangle(image_np_with_detections, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), 2, 1)
cv2.imshow('object detection',image_np_with_detections)
cv2.waitKey(0)
print('Done')
cv2.destroyAllWindows()
"""

boxes = detections['detection_boxes']
# get all boxes from an array
max_boxes_to_draw = boxes.shape[0]
# get scores to get a threshold
scores = detections['detection_scores']
# this is set as a default but feel free to adjust it to your needs
empty=0
full=0
# iterate over all objects found
for i in range(min(max_boxes_to_draw, boxes.shape[0])):

    if scores is None or scores[i] > MIN_CONF_THRESH:
        # boxes[i] is the box which will be drawn
        class_name = category_index[detections['detection_classes'][i]]['name']
        print ("This box is gonna get used", boxes[i], detections['detection_classes'][i])  #box_coordinate,box_label
        (left, right, top, bottom) = (boxes[i][1] * im_width,boxes[i][3]  * im_width,
                                      boxes[i][0] * im_height, boxes[i][2] * im_height)
        #print(left, right, top, bottom) #left,right,top,bottom
        cv2.putText(image_np_with_detections,str(i) , (int(right)-20, int(bottom)), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), thickness=3,fontScale=1)

        if(detections['detection_classes'][i]==1):
            empty=empty+1
        elif (detections['detection_classes'][i] ==2):
            full=full+1
#print(scores)
total="total:"+str(empty+full)+"/empty:"+str(empty)
#print(total)
image_np_with_detections=cv2.cvtColor(image_np_with_detections,cv2.COLOR_BGR2RGB)
cv2.putText(image_np_with_detections,total,(20,40),cv2.FONT_HERSHEY_SIMPLEX,color=(0,0,0),thickness=3,fontScale=1)
cv2.imshow('object detection',image_np_with_detections)
key = cv2.waitKey(0) & 0xff
if key in range(48, 52): # 0~2 숫자 입력   ---⑥
    trackerIdx = key-48     # 선택한 숫자로 트랙커 인덱스 수정
    if (detections['detection_classes'][trackerIdx] == 1):
        tkinter.messagebox.showinfo("메세지","예약완료")
    elif (detections['detection_classes'][trackerIdx] == 2):
        tkinter.messagebox.showinfo("메세지","좌석확인바람")
print('Done')
cv2.destroyAllWindows()
"""