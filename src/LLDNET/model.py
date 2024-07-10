import os
import cv2
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Activation,AveragePooling2D ,ReLU 
                                     ,Add,Dense,Reshape,Multiply, Dropout,
                                       UpSampling2D,Conv2D,MaxPooling2D,
                                       concatenate,add,GlobalAveragePooling2D,
                                       GlobalMaxPool2D,Multiply)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

