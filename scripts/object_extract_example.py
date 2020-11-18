import cv2
import numpy as np

cap = cv2.VideoCapture(0) #primary webcam
while True:
    _, frame = cap.read()

    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)

    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # for c in contours:
    #     rect = cv2.minAreaRect(cnt)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     im = cv2.drawContours(im,[box],0,(0,0,255),2)
    #     # x,y,w,h = cv2.boundingRect(c)
    #     # img = cv2.rectangle(edged,(x,y),(x+w,y+h),(0,255,0),2)
    #     # cv2.imshow('Canny Edges After Contouring', img)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 3:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    # Display the resulting frame
    cv2.imshow('frame',frame)

    #cv2.imshow('Canny Edges After Contouring', edged)
    #cv2.imshow("Frame",frame)



    if cv2.waitKey(1) & 0xFF == ord('q'): # hit "q to kill window"
        break
cap.release()
cv2.destroyAllWindows()




# shape approx https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
