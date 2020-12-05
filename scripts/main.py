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


class ImageExtractor():
    """ This ImageExtractor class converts a ROS video stream
        to OpenCV format, and then identifies images that can
        be fed into an image classifier. """

    def __init__(self, image_topic):
        rospy.init_node('sign_finder')

        self.cv_image = None        # the latest image from the camera
        self.bridge = CvBridge()    # used to convert ROS messages to OpenCV

        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.contour_image = None
        self.threshold_image = None
        self.roi_image = None
        self.rectangle_image = None

        rospy.Subscriber(image_topic, Image, self.get_image)

    def get_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def sign_localizer(self, img):
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (200, 0, 0)

        self.contour_image = np.zeros(img.shape)
        blurred_frame = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
        self.threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 73, 5)
        contours, hierarchy = cv2.findContours(self.threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.contour_image, contours, -1, (0,255,0), 3)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        self.x,self.y,self.w,self.h = cv2.boundingRect(cnt)
        self.rectangle_image = cv2.rectangle(img,(self.x,self.y),(self.x+self.w,self.y+self.h),(0,255,0),3)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(20,40,255),2)

    def save_image(self, img):
        self.roi_image = img[self.y:self.y+self.h, self.x:self.x+self.w]
        cv2.imwrite("roi.png", self.roi_image)

class SignRecognition():
    def __init__(self, downloaded_data_path, image_path):
        self.img_extractor = ImageExtractor("/camera/image_raw")
        self.cnn = ConvNeuralNet()
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

    def load_data(self, image_dir, categories,
                  img_h, img_w, batch, grayscale):

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

        class_names = self.train_ds.class_names

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #keep images in memory
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    def train_cnn(self):
        self.model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # layers.Conv2D(32, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Conv2D(64, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(self.N_CLASSES)
        ])

        self.model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

        self.model.summary()

        # self.history = self.model.fit(
        # self.train_ds,
        # validation_data=self.val_ds,
        # epochs=self.epochs
        # )

    def detect_image();
        pass

    def run(self):
        r = rospy.Rate(5)
        self.load_data(self.image_path, self.selected_categories, 64, 64, 32, 0)
        self.train_cnn()
        while not rospy.is_shutdown():
            if not self.img_extractor.cv_image is None:
                self.img_extractor.sign_localizer(self.img_extractor.cv_image)
                #visualize video feedq
                cv2.imshow('binary_window', self.img_extractor.cv_image)
                cv2.imshow('Threshold', self.img_extractor.threshold_image)
                cv2.imshow('Contours', self.img_extractor.contour_image)
                self.img_extractor.save_image(self.img_extractor.cv_image)
                cv2.waitKey(5)
            r.sleep()
            if cv2.waitKey(1) & 0xFF == ord('q'): #kill open CV windows
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sign_recognition = SignRecognition("/home/abbymfry/Desktop/chinese_traffic_signs/", "/home/abbymfry/catkin_ws/src/computer_vision/scripts/images")
    sign_recognition.run()


    ### next step is to predict with ROI using model.predict, need to make sure it is right pixel sizing

    # TODO
    # Figure out sign detected --> sign identified pipeline
    # i.e. CNN should be pre-trained, and we want to use it!
    # Save gazebo world as .launch file
    # Make one more sign model (new texture)
    # Need to hone video shape/sign detection situation for Gazebo world
    # i.e. maybe play with threshold
    # MVP: try to get sign recognition working for basic gazebo world by next Wednesday,
    # and plan to improve it by Dec 13. Then finish documenting!
    # Talk about our 3 classes - think about architecture
