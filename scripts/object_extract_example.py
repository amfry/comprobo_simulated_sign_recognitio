import cv2
import numpy as np

cap = cv2.VideoCapture(0) #primary webcam
while True:
    _, frame = cap.read()

    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)

    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)
        epsilon = 0.1*cv2.arcLength(c,True)
        approx_vertices = cv2.approxPolyDP(c,epsilon,True)
        # if len(approx) == 3:
        #     x,y,w,h = cv2.boundingRect(c)
        #     image = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        #     cv2.putText(image, 'Triangle', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        #     cv2.imshow('frame',frame)
        # if len(approx) == 4:
        print(approx_vertices)
        if area > 50:
            x,y,w,h = cv2.boundingRect(c)
            image = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(image, 'Rectangle: ' + str(area), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.imshow('frame',frame)



    # Display the resulting frame
    #cv2.imshow('frame',frame)

    #cv2.imshow('Canny Edges After Contouring', edged)
    #cv2.imshow("Frame",frame)



    if cv2.waitKey(1) & 0xFF == ord('q'): # hit "q to kill window"
        break
cap.release()
cv2.destroyAllWindows()




# shape approx https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
