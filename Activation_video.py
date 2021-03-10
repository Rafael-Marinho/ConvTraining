from cv2 import VideoWriter, VideoWriter_fourcc, VideoCapture, dnn, FONT_HERSHEY_PLAIN, rectangle, putText, imshow, waitKey, destroyAllWindows
from numpy import argmax


# Load the convolutional neural network and its architecture file:
net = dnn.readNet("Networks/Model.weights", "Networks/Model.cfg")

# Load the classes:
classes = []
with open("Networks/Model.names", 'r') as f:
    classes = f.read().splitlines()

# Define font, calls video stream to activate the neural network and set the configurations to create a output video:
cap = VideoCapture("input.mp4")
out = VideoWriter('output.avi', VideoWriter_fourcc(*"mp4v"), 20, (int(cap.get(3)), int(cap.get(4))))

# All the video stream and instances detection, draw and display stuff:
while (1 < 2):

    # Execution variables, pointers and stuff:
    _, img = cap.read()

    try:
        height, width, _ = img.shape
    except(AttributeError):
        break
    net.setInput(dnn.blobFromImage(img, (1 / 255), (415, 416), (0, 0, 0), swapRB=True, crop=False))
    boxes = []
    confidences = []
    class_ids = []

    # Detection and storage of objects instances:
    for output in net.forward(net.getUnconnectedOutLayersNames()):
        for detection in output:
            scores = detection[5:]
            class_id = argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Draw the instances rectangles, classes and confidences on the frame:
    indexes = dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if (len(indexes) > 0):
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            rectangle(img, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
            putText(img, (label + ' ' + confidence), (x, (y + 20)), FONT_HERSHEY_PLAIN, 1, (200, 0, 180), 2)

    # Display the frames and the drawed content:
    imshow("DATA", img)
    out.write(img)

    # Press Esc to abort, if you like to:
    key = waitKey(1)
    if (key == 27):
        break

# Closes the pointers of input and output, closing all the windows at the end.
cap.release()
out.release()
destroyAllWindows()
