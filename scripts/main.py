#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image

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

    def run(self):
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                self.sign_localizer(self.cv_image)
                #visualize video feedq
                cv2.imshow('binary_window', self.cv_image)
                cv2.imshow('Threshold', self.threshold_image)
                cv2.imshow('Contours', self.contour_image)
                self.save_image(self.cv_image)
                cv2.waitKey(5)
            r.sleep()
            if cv2.waitKey(1) & 0xFF == ord('q'): #kill open CV windows
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    node = ImageExtractor("/camera/image_raw")
    node.run()

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
