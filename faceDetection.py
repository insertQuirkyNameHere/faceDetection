#done using classifiers
#classifiers are trained with 10s of 1000s of images
#opencv comes with some classifiers
#Local Binary PAtterns, and Haar Cascades are examples.
import cv2 as cv

videoFeed = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    isReceiving, frame = videoFeed.read()
    cv.imshow("Frame", frame)
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray", gray)

    blurred = cv.GaussianBlur(gray, (13,13), 0)
    cv.imshow("Blur", blurred)

    haarCascade = cv.CascadeClassifier("haar_face.xml")
    faceRect = haarCascade.detectMultiScale(blurred, scaleFactor=1.1, minNeighbors=7)

    for(x1, y1, x2, y2) in faceRect:
        cv.rectangle(frame, (x1,y1), (x1+x2, y1+y2), (0, 255, 0), 2)

    cv.imshow("Detection", frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

videoFeed.release()
cv.destroyAllWindows()

""" img = cv.imread("Python/Opencv/face.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Original Image", img)
cv.imshow("Grayscale Image", gray)

haarCascade = cv.CascadeClassifier("Python/Opencv/haar_face.xml")

faceRect = haarCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print(len(faceRect), "faces found")

for(x1, y1, x2, y2) in faceRect:
    cv.rectangle(img, (x1,y1), (x1+x2, y1+y2), (0, 255, 0), 2)

cv.imshow("faces detected", img)
cv.waitKey(0)
cv.destroyAllWindows() """