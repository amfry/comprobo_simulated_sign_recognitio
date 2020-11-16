import cv2
cap = cv2.VideoCapture(0) #primary webcam
while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 70)
    ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow('Video feed', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'): # hit "q to kill window"
        break
cap.release()
cv2.destroyAllWindows()


# https://pysource.com/2018/12/29/real-time-shape-detection-opencv-with-python-3/
