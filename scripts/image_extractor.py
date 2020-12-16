#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from sign_classification import ConvNeuralNet
import time



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
        """Takes in video feed and saves to class attributes a region of
        interest and True/False flag for if a sign was detected"""
        img = img[0:300, 0:700]  #only look at top portion of video feed
        font = cv2.FONT_HERSHEY_COMPLEX
        self.contour_image = np.zeros(img.shape)
        blurred_frame = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
        self.threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 73, 5)
        contours, hierarchy = cv2.findContours(self.threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.contour_image, contours, -1, (0,255,0), 3)
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

    def save_image(self, img):
        """Saves ROI to current working directory"""
        self.roi_image = self.clean_image[self.y-25:self.y+self.h+25, self.x-25:self.x+self.w+25]
        if (self.y < 50):
            self.sign_reached == True
        try:
            cv2.imwrite("roi.png", self.roi_image)
        except:
            pass
