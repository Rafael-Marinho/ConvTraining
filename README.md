# ConvTraining
This project has been made in purpose to simplify the process of Darknet / YOLO convolutional network training. Go ahead, it's open-source, copyleft and you must feel free to hack it at your desire.

Cheers to The AI Guy (https://github.com/theAIGuysCode).

#### You will need:
- Darknet and YOLOv4 (other versions must work too, I don't know);
- Python3 (Python2 must work too but it's dead since january 1st 2020, go ahead and bury it and go to Python3 once for all);
- Cmake;
- OpenCV (4.4 or higher);
- NumPy.

The OpenCV and NumPy are Python packages. Once you have the Python installed, you can download and install them with:

```
pip install opencv-python numpy
```

## Data labelling:
You can do it anyway you like; I used LabelImg (https://github.com/tzutalin/labelImg) but there are other, more automatic methods.
#### ALL THE IMAGES MUST BE .JPG FORMAT.

## Network configuration and stuff:
Once you labelled the images:
1) Put all the images and their .txt files with classes coordinates at Model/data/obj (replace the content already there);
2) Do the same with the classes.txt and classes.names;
3) Change the parameter classes at the Model/data/obj/obj.data to the number of classes on the classes.txt;
4) Insert images and .txt files at Model/data/test, they must be from the same nature of the ones at Model/data/obj (they can be even a subset from them), to be used for the validation during the training process.
5) At Model/data/obj.names, insert the same content of classes.names but with "\_" instead of " " (replace the content already there);
6) Run the generate_test.py, generate_train.py with Python (many thanks to The AI Guy for providing us these scripts);
7) Run the generate_cfg.py with Python (many thanks to The AI Guy for providing me the instructions to code this script);
8) Add a convolutional neural network to be used as a base (you can get a good one here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137)

## Network training:
On the same folder where is the Model dir, install the Darknet (https://pjreddie.com/darknet/install/). Do it with CUDA resources if you can.

(Seriously, the training with no dedicated GPGPU may take **months** to reach a good couple of weights).

If you are a f...ed up guy like me, you can use a good GPGPU from Google Colab for free (twelve hours for day at most), and if you live in USA, you can even rent processing power as an IaaS. You can access it at https://colab.research.google.com/; and you will need a Google account for it, of course.

#### Following these commands, you can download, configure and install Darknet with CUDA resources at Colab (remove the ! and % if you gonna run it into your machine instead):

```
!git clone https://github.com/AlexeyAB/darknet

%cd darknet
!sed -i "s/OPENCV=0/OPENCV=1/" Makefile
!sed -i "s/GPU=0/GPU=1/" Makefile
!sed -i "s/CUDNN=0/CUDNN=1/" Makefile
!sed -i "s/CUDNN_HALF=0/CUDNN_HALF=1/" Makefile

!make
```

#### (IT WILL ONLY WORK IF THE NOTEBOOK IS SET TO USE GPU)

If you are using the Colab, you need to upload the files. I recommend you to upload them in your Drive (you're already using your Google account after all, isn't?). So, uploading the files at the root of your Drive, you just need to access it at the Colab by the following command:

```
%cd "/content/drive/My Drive/Model/"
```

Anyway, the command to train the convolutional neural network follows the pattern:

_./darknet detector train [obj.data] [CFG file] [base training file] -dont_show -map_

So, following the logic of the directories tree and considering you downloaded the recommended base training file (yolov4.conv.137), the command is:

```
! ./../../../darknet/darknet detector train data/obj/obj.data Model.cfg yolov4..conv.137 -dont_show -map
```

... AND that's it. Just let the computer train the neural network for a couple of hours (or even days, it's up to you). Use that time to sleep, work, play a good grunge song, learn Russian, meditate, live, love or think about your Ph.D.

## Network activation:
So you trained the weights for a long time. Make it worth by accessing the backup directory and take the weight files (the Model_best.weights is the guy you has been waiting for).

Take the Model_best.weights, rename it to "Model.weights" and move it to the Networks directory (replace the one there). Also, move the Model.cfg and Model.names to the same directory.

After that, take the video which you want to activate your brand new convolutional neural network, put at the root of project and rename it to "input.mp4".

And then sit down straight, correct your posture, take a deep breath and run the Activate.py with Python.

At this point, pay attention at the classes detection performance and not the FPS rate -- after all, you didn't trained your hardware but your convolutional neural network.


### That's it. Regards and keep rocking, guys.
