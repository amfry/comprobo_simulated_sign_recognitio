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

        rospy.Subscriber(image_topic, Image, self.get_image)

        cv2.namedWindow('binary_image')

    def get_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.binary_image = cv2.inRange(self.cv_image, (128,128,128), (255,255,255))

    def sign_localizer(self, img):
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (200, 0, 0)

        img_contours = np.zeros(img.shape)
        blurred_frame = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)

        #_, threshold = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 73, 5)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            area = cv2.contourArea(cnt)
            if 1000 < area < 100000:
                cv2.drawContours(img, [approx], 0, (0,55,55), 5)
                if len(approx) == 3:
                    cv2.putText(img, "Triangle", (x, y), font, 1, color)
                elif len(approx) == 4:
                    cv2.putText(img, "Rectangle", (x, y), font, 1, color)
                elif len(approx) == 5:
                    cv2.putText(img, "Pentagon", (x, y), font, 1, color)
                elif len(approx) == 6:
                    cv2.putText(img, "Hexagon", (x, y), font, 1, color)
                elif  15< len(approx) < 50:
                    cv2.putText(img, "Circle", (x, y), font, 1, color)

        cv2.imshow("shapes", img)
        cv2.imshow("Threshold", thresh)

    def run(self):
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                self.sign_localizer(self.cv_image)
                print(self.cv_image.shape)
                cv2.imshow('binary_window', self.cv_image)
                cv2.waitKey(5)

            # start out not issuing any motor commands
            r.sleep()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Hello World")
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

