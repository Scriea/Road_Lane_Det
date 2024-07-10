
import numpy as np
import cv2
#from scipy.misc import imresize
import time
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import argparse
import os 

def dsc(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def dice_loss(y_true, y_pred):
        return 1.0 - dsc(y_true, y_pred)


def IOU(y_true, y_pred):

        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
    
        thresh = 0.5

        y_true = K.cast(K.greater_equal(y_true, thresh), 'float32')
        y_pred = K.cast(K.greater_equal(y_pred, thresh), 'float32')

        union = K.sum(K.maximum(y_true, y_pred)) + K.epsilon()
        intersection = K.sum(K.minimum(y_true, y_pred)) + K.epsilon()

        iou = intersection/union

        return iou

parser = argparse.ArgumentParser()
parser.add_argument('--data', required = True)
parser.add_argument('--type', type = int, required = True)


args = parser.parse_args()


model = load_model('/home/screa/Desktop/Road_Lane_Det/models/LLDNet.h5', custom_objects={'dice_loss':dice_loss,'IOU':IOU,'dsc':dsc,'precision_m':precision_m, 'recall_m':recall_m, 'f1_m':f1_m})

# Class to average lanes with
class Lanes():
    def _init_(self):
        self.recent_fit = []
        self.avg_fit = []


if args.type ==1:            ## Images
    
    lanes = Lanes()
    image = cv2.imread(args.data)
    small_img = cv2.resize(image, (160,80))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]
    prediction = model.predict(small_img)[0] * 255

    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    lane_impr = cv2.morphologyEx(lane_drawn, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    #            cv2.imshow('swef', lane_impr)
    # Re-size to match the original image
    lane_image = cv2.resize(lane_drawn, (640,480))

    image = cv2.resize(image, (640,480))
    lane_image = lane_image.astype(np.uint8)

    result = cv2.addWeighted(image, 1, lane_image, 1, 0)


    cv2.imshow('r',result)
    cv2.imwrite('..' +os.sep + 'results'+ os.sep+'Output.png',result)
    cv2.waitKey(0) == ord('q')
    cv2.destroyAllWindows()


elif args.type==2:           ## Video

    video_capture = cv2.VideoCapture(args.data)

    video_capture.set(3, 640)
    video_capture.set(4, 480)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    #print("a:",fps)

      
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    out=cv2.VideoWriter('..' +os.sep + 'results'+ os.sep+'Output.avi' , fourcc , 20.0 , (640,480))
    count = 0
    time_taken = []
        
    while(video_capture.isOpened()):
       lanes = Lanes()

    # Capture the frames
       ret,image = video_capture.read()

       if not ret:
           print("Can't receive frame (stream end?). Exiting ...")
           break
       
       if ret:

               image1 = image.copy()
               image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
               image = cv2.equalizeHist(image)
               image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
               kernel1 = np.array([[0, -1, 0],
                                [-1, 6, -1],
                                [0, -1, 0]])
               image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
               small_img = cv2.resize(image, (160,80))
               small_img = np.array(small_img)
             
               small_img = small_img[None,:,:,:]
               start_time=time.time()
               prediction = model.predict(small_img)[0] * 255
               total_time = time.time() - start_time
               
               fps = 1/(time.time()-start_time)
               
               fps = int(fps)
               print(fps)
              
               lanes.recent_fit.append(prediction)
         # Only using last five for average
               if len(lanes.recent_fit) > 5:
                  lanes.recent_fit = lanes.recent_fit[1:]

        # Calculate average detection
               lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

        # Generate empty R & B color dimensions, stack with G
               blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
               lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
               
               lane_impr = cv2.morphologyEx(lane_drawn, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # Re-size to match the original image
               lane_image = cv2.resize(lane_drawn, (640,480))

               image1 = cv2.resize(image1, (640,480))
               lane_image = lane_image.astype(np.uint8)

               result = cv2.addWeighted(image1, 1, lane_image, 1, 0)
               
               out.write(result)
               cv2.imshow('r',result)
               
      
      
               if cv2.waitKey(1) & 0xFF == ord('q'):

                     break


    video_capture.release()
    out.release()
    cv2.destroyAllWindows()