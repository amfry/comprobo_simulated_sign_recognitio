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

class CreateCNN():
    def __init__(self, downloaded_data_path, image_path):
        # self.img_extractor = ImageExtractor("/camera/image_raw")
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
        self.loaded_model = None

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

        self.history = self.model.fit(
        self.train_ds,
        validation_data=self.val_ds,
        epochs=20
        )

        print(self.history.history['accuracy'])

        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

if __name__ == '__main__':
    #model_creator = CreateCNN("/home/vscheyer/Desktop/traffic_sign_dataset/", "/home/vscheyer/catkin_ws/src/computer_vision/scripts/images")
    model_creator = CreateCNN("/home/abbymfry/Desktop/chinese_traffic_signs/", "/home/abbymfry/catkin_ws/src/computer_vision/scripts/images")
    model_creator.load_data(model_creator.image_path, model_creator.selected_categories, 64, 64, 32, 0)
    model_creator.train_cnn()
