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
        self.sign_flag = False
        # self.sign_flag_temp = False
        self.sign_reached = False

        rospy.Subscriber(image_topic, Image, self.get_image)

    def get_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.clean_image = self.cv_image

    def sign_localizer(self, img):
        img = img[0:300, 0:700]  #only look at top portion of video feed
        font = cv2.FONT_HERSHEY_COMPLEX
        self.contour_image = np.zeros(img.shape)
        blurred_frame = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
        self.threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 73, 5)
        contours, hierarchy = cv2.findContours(self.threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.contour_image, contours, -1, (0,255,0), 3)
        # areas = [cv2.contourArea(c) for c in contours]  for pulling largest area contour
        # max_index = np.argmax(areas)
        # cnt=contours[max_index]
        # approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        interest_area = []
        interest_cnt = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            area = cv2.contourArea(c)
            if 20000 > area > 1000:
                if len(approx) > 4:
                    interest_area.append(area)
                    interest_cnt.append(c)

        self.sign_flag_temp = self.sign_flag

        if len(interest_cnt) ==  0:  #no regions detected
            self.x = None
            self.y = None
            self.w = None
            self.h = None
            self.rectangle_image = None
            self.sign_flag = False
            print("No Sign Detected")
        else:
            max_area_index = np.argmax(interest_area)
            cnt = interest_cnt[max_area_index]
            self.x,self.y,self.w,self.h = cv2.boundingRect(cnt)
            self.rectangle_image = cv2.rectangle(img,(self.x-30,self.y-30),(self.x+self.w+30,self.y+self.h+30),(0,255,0),3)
            self.sign_flag = True
            # print("Sign Detected!")

        # if (self.sign_flag_temp != self.sign_flag):
        #     self.new_sign = True

    def save_image(self, img):
        self.roi_image = self.clean_image[self.y-25:self.y+self.h+25, self.x-25:self.x+self.w+25]
        # print("ROI Y:")
        # print(self.y)
        if (self.y < 50):
            self.sign_reached == True
        try:
            cv2.imwrite("roi.png", self.roi_image)
        except:
            pass

class RobotMotion():
    def __init__(self):
        #rospy.init_node('robo_motion')
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.line_vel = 0.15
        self.ang_vel = 0.25
        self.expected_sign = 52
        self.detected_sign = None

    def starter_motion(self):
        self.pub.publish(Twist(linear=Vector3(x=self.line_vel, y=0)))

    def stop(self):
        self.pub.publish(Twist(linear=Vector3(x=0, y=0), angular=Vector3(z=0)))
        time.sleep(2)

    def turn_right(self):
        self.pub.publish(Twist(linear=Vector3(x=0, y=0), angular=Vector3(z=-0.55)))
        time.sleep(3)


class SignRecognition():
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

    def train_cnn(self):
        self.model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(57)
        ])

        self.model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

        self.model.summary()

        # self.history = self.model.fit(
        # self.train_ds,
        # validation_data=self.val_ds,
        # epochs=20
        # )

    def load_cnn(self):
        # load json and create model
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
        # print("%s: %.2f%%" % (self.loaded_model.metrics_names[1], score[1]*100))


    def detect_image(self):
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

        print(
        "Class {} with {:.2f}"
        .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )

    def run(self):
        r = rospy.Rate(5)
        self.load_data(self.image_path, self.selected_categories, 64, 64, 32, 0)
        # self.train_cnn()
        self.load_cnn()
        # if not rospy.is_shutdown():
        #     self.robo_motion.starter_motion()
        while not rospy.is_shutdown():
            if not self.img_extractor.cv_image is None:
                self.img_extractor.sign_localizer(self.img_extractor.cv_image)
                #visualize video feedq
                #cv2.imshow('Video Feed', self.img_extractor.cv_image)
                cv2.imshow('Threshold', self.img_extractor.threshold_image)
                cv2.imshow('Contours', self.img_extractor.contour_image)
                # print("NEW SIGN:")
                # print(self.new_sign)
                print(type(self.confidence))
                if (self.img_extractor.sign_flag == True):
                    self.img_extractor.save_image(self.img_extractor.rectangle_image)
                    cv2.imshow('Clean Image', self.img_extractor.clean_image)
                    self.detect_image()
                    self.robo_motion.detected_sign = self.prediction
                    # print("detected sign class:")
                    # print(self.robo_motion.detected_sign)
                if ((self.img_extractor.sign_flag) and (self.img_extractor.y < 50) and (self.new_sign) and (self.confidence > 0.98)):
                    # print("STOPPPPPPPPP")
                    # if (self.prediction == 52):
                    self.robo_motion.stop()
                    # if (self.prediction == 24):
                        # self.robo_motion.turn_right()
                    self.new_sign = False

                self.robo_motion.starter_motion()
                # else:
                #     self.robo_motion.starter_motion()
                cv2.waitKey(5)
            r.sleep()
            if cv2.waitKey(1) & 0xFF == ord('q'): #kill open CV windows
                break
        cv2.destroyAllWindows()




if __name__ == '__main__':
    ## sign_recognition = SignRecognition("/home/vscheyer/Desktop/traffic_sign_dataset/", "/home/vscheyer/catkin_ws/src/computer_vision/scripts/images")
    sign_recognition = SignRecognition("//home/vscheyer/Desktop/traffic_sign_dataset/", "//home/vscheyer/catkin_ws/src/computer_vision/scripts/images")
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
