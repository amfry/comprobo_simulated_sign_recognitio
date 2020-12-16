#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from sign_classification import ConvNeuralNet
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
import os
from tensorflow.keras.models import model_from_json
from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, rotation_matrix, quaternion_from_matrix
import time
from geometry_msgs.msg import Twist, Vector3

from image_extractor import ImageExtractor
from robot_motion import RobotMotion

class SignRecognition():
    """Identfies simulated road sign using a CNN and commands simulated neato to take
    appropriate action

    downloaded_data_path (str): path to training data
    image path (str): path to directory where training data gets sorted into folders by class
    """
    def __init__(self, downloaded_data_path, image_path):
        self.img_extractor = ImageExtractor("/camera/image_raw")
        self.cnn = ConvNeuralNet()
        self.robo_motion = RobotMotion()
        self.downloaded_data_path = downloaded_data_path
        self.image_path = image_path
        self.selected_categories = [range(1,57)]
        self.epochs = 25
        self.model = None
        self.history = None
        self.train_ds = None
        self.N_CLASSES = len(self.selected_categories)
        self.IMG_HEIGHT = 64
        self.IMG_WIDTH = 64
        self.CHANNELS = 3
        self.val_ds = None
        self.batch_size = 32
        self.loaded_model = None
        self.confidence = None
        self.prediction = 0
        self.prediction_temp = 1
        self.class_names = None
        self.new_sign = False

    def load_data(self, image_dir, categories,
                  img_h, img_w, batch, grayscale):

        """load data from training set to get self.val_ds"""

        self.N_CLASSES = len(categories) # CHANGE HERE, total number of classes
        self.IMG_HEIGHT = img_h # CHANGE HERE, the image height to be resized to
        self.IMG_WIDTH = img_w # CHANGE HERE, the image width to be resized to
        if (grayscale == 0):
            self.CHANNELS = 3 # The 3 color channels, change to 1 if grayscale
        else:
            self.CHANNELS = 1
        self.batch_size = batch

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
        batch_size=self.batch_size)

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
        batch_size=self.batch_size)

        self.class_names = self.train_ds.class_names

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #keep images in memory
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    def load_cnn(self):
        """ load json and create model """
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        self.loaded_model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

        score = self.loaded_model.evaluate(self.val_ds, verbose=0)

    def detect_image(self):
        """uses loaded CNN to idnetify what is in ROI"""
        self.current_dir = os.getcwd()
        img = keras.preprocessing.image.load_img(self.current_dir + "/roi.png", target_size=(64, 64))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        self.prediction_temp = self.prediction

        predictions = self.loaded_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        self.prediction = self.class_names[np.argmax(score)]
        self.confidence = np.max(score)

        if (self.prediction_temp != self.prediction):
            self.new_sign = True

    def run(self):
        """the main loop for sign detection"""
        r = rospy.Rate(5)
        self.load_data(self.image_path, self.selected_categories, 64, 64, 32, 0)
        self.load_cnn()
        while not rospy.is_shutdown():
            if not self.img_extractor.cv_image is None:
                self.img_extractor.sign_localizer(self.img_extractor.cv_image)
                cv2.imshow('Threshold', self.img_extractor.threshold_image)
                cv2.imshow('Contours', self.img_extractor.contour_image)
                if (self.img_extractor.sign_flag == True):
                    self.img_extractor.save_image(self.img_extractor.rectangle_image)
                    cv2.imshow('Clean Image', self.img_extractor.clean_image)
                    self.detect_image()
                    self.robo_motion.detected_sign = self.prediction
                if ((self.img_extractor.sign_flag) and (self.img_extractor.y < 50) and (self.new_sign) and (self.confidence > 0.98)):
                    if (self.prediction == 'class_52'):
                        print('STOP')
                        self.robo_motion.stop()
                    if (self.prediction == 'class_24'):
                        print('TURN RIGHT')
                        self.robo_motion.turn_right()
                    if (self.prediction == 'class_22'):
                        print('TURN LEFT')
                        self.robo_motion.turn_left()
                    self.new_sign = False

                self.robo_motion.starter_motion()

                cv2.waitKey(5)
            r.sleep()
            if cv2.waitKey(1) & 0xFF == ord('q'): #kill open CV windows
                break
        cv2.destroyAllWindows()




if __name__ == '__main__':
    sign_recognition = SignRecognition("/home/vscheyer/Desktop/traffic_sign_dataset/", "/home/vscheyer/catkin_ws/src/computer_vision/scripts/images")
    #sign_recognition = SignRecognition("//home/vscheyer/Desktop/traffic_sign_dataset/", "//home/vscheyer/catkin_ws/src/computer_vision/scripts/images")
    sign_recognition.run()
