# ConvTraining
This project has been made in purpose to simplify the process of Darknet / YOLO convolutional network training. Go ahead, it's open-source, copyleft and you must feel free to hack it at your desire.

Cheers to The AI Guy (https://github.com/theAIGuysCode).

#### You will need:
- Darknet and YOLOv4 (other versions must work too, I don't know);
- Python3 (Python2 must work too but it's dead since january 1st 2020, go ahead and bury it and go to Python3 once for all);
- Cmake;
- OpenCV (4.4 or higher);

## Data labelling:
You can do it anyway you like; I used LabelImg (https://github.com/tzutalin/labelImg) but there are other, more automatic methods.

## Network configuration and stuff:
Once you labelled the images:
1) Put all the images and their .txt files with classes coordinates at Model/data/obj (replace the content already there);
2) Do the same with the classes.txt and classes.names;
3) Change the parameter classes at the Model/data/obj/obj.data to the number of classes on the classes.txt;
4) At Model/data/obj.names, insert the same content of classes.names but with "\_" instead of "\ " (replace the content already there);
5) Run the generate_test.py, generate_train.py with Python;
6) Run the generate_cfg.py with Python.

## Network training:

## Network activation:
