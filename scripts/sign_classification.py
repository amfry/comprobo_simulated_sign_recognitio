import os
# import cv2
from dataset_organizing import FileOrganizing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class ConvNeuralNet():
    def __init__(self):
      self.train_ds = None 
      self.val_ds = None
      # Image Parameters
      self.N_CLASSES = 3
      self.IMG_HEIGHT = 64
      self.IMG_WIDTH = 64
      self.CHANNELS = 3
      self.batch_size = 32
      self.history = None
      self.epochs = 25

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
      print(class_names)

      AUTOTUNE = tf.data.experimental.AUTOTUNE

      self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #keep images in memory
      self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

      normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)  #rescale RGB valyes

    def make_model(self, epochs):
      self.epochs = epochs
      model = Sequential([
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

      model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

      model.summary()

      self.history = model.fit(
        self.train_ds,
        validation_data=self.val_ds,
        epochs=self.epochs
      )

      print("PREDICTION:")
      print(model.predict('/home/vscheyer/catkin_ws/src/computer_vision/scripts/images/005_1_0001.png'))      

    def analyze_results(self):
      acc = self.history.history['accuracy']
      val_acc = self.history.history['val_accuracy']

      loss = self.history.history['loss']
      val_loss = self.history.history['val_loss']

      for i in range (0,len(acc)):
        acc[i] = acc[i]*100
        val_acc[i] = val_acc[i]*100

      epochs_range = range(self.epochs)

      plt.figure(figsize=(8, 8))
      plt.subplot(1, 2, 1)
      plt.plot(epochs_range, acc, label='Training Accuracy')
      plt.plot(epochs_range, val_acc, label='Validation Accuracy')
      plt.legend(loc='lower right')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy (%)')

      plt.subplot(1, 2, 2)
      plt.plot(epochs_range, loss, label='Training')
      plt.plot(epochs_range, val_loss, label='Validation')
      plt.legend(loc='upper right')
      plt.title('Training and Validation Loss')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.show()



    

    
