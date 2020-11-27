import cv2
import numpy as np

cap = cv2.VideoCapture(0) #primary webcam
font = cv2.FONT_HERSHEY_COMPLEX
color = (200, 0, 0)

while True:
    _, frame = cap.read()
    img_contours = np.zeros(frame.shape)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
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
            cv2.drawContours(frame, [approx], 0, (0,55,55), 5)
            if len(approx) == 3:
                cv2.putText(frame, "Triangle", (x, y), font, 1, color)
            elif len(approx) == 4:
                cv2.putText(frame, "Rectangle", (x, y), font, 1, color)
            elif len(approx) == 5:
                cv2.putText(frame, "Pentagon", (x, y), font, 1, color)
            elif len(approx) == 6:
                cv2.putText(frame, "Hexagon", (x, y), font, 1, color)
            elif  15< len(approx) < 50:
                cv2.putText(frame, "Circle", (x, y), font, 1, color)

    cv2.imshow("shapes", frame)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'): # hit "q to kill window"
        break
cap.release()
cv2.destroyAllWindows()
