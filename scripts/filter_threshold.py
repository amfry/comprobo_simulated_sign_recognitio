import cv2
import numpy as np
from matplotlib import pyplot as plt

path = "/home/abbymfry/Desktop/image.png"
path = "/home/abbymfry/Desktop/next.png"

img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img, (5, 5), 0)

def range_block_size(img):
    images = []
    titles = []
    for i in range(3,256, 2):
        print(i)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, i, 2)
        images.append(thresh)
        titles.append("block_" + str(i))
    for i in range(0,len(images)):
        plt.subplot(9,15,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

if __name__ == "__main__":
    range_block_size(img)
