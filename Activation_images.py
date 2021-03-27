from os import path, getcwd, chdir, listdir
from cv2 import imread, imwrite, dnn, FONT_HERSHEY_PLAIN, rectangle, putText, imshow, waitKey, destroyAllWindows
from numpy import argmax


# Load list of images on activation path:
image_files = []
chdir(path.join("input"))
for filename in listdir(getcwd()):
    if filename.endswith(".jpg"):
        image_files.append(filename)
chdir("..")

# Load the convolutional neural network and its architecture file:
net = dnn.readNet("Networks/Model.weights", "Networks/Model.cfg")

# Load the classes:
classes = []
with open("Networks/Model.names", 'r') as f:
    classes = f.read().splitlines()

# Run the activation in every JPG file inside the activation path:
for image in image_files:
    # Load image:
    img = imread("input/" + image)

    # Execution variables, pointers and stuff:
    height, width, _ = img.shape

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
            if (confidence > 0.5):
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

    # Save the output image:
    imwrite(("output/" + image), img)
