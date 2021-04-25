# ConvTraining
This project has been made in purpose to simplify the process of Darknet / YOLO convolutional network training. Go ahead, it's open-source, copyleft and you must feel free to hack it at your desire.

#### You will need:
- Darknet and YOLOv4 (other versions must work too, I did not test on YOLOv5 yet);
- Python3 (Python2 must work too but it's dead since january 1st 2020, go ahead and bury it and go to Python3 once for all);
- Cmake;
- OpenCV (4.4 or higher);
- NumPy.

The OpenCV and NumPy are Python packages. Once you have the Python installed, you can download and install them with:

```
pip install opencv-python numpy
```

## Data labelling:
You can do it anyway you like; I used LabelImg (https://github.com/tzutalin/labelImg) but there are others, more automatic methods.
#### ALL THE IMAGES MUST BE .JPG FORMAT.
Why only .jpg? The generate_stuff.py will look only for jpg files, once its a format Darknet has no problems to deal with. I know it doesn't work with .tif files as well; you can hack the generate_stuff.py to accept another formats for experimentation (there are two lines at the very end of the file, just use a Ctrl+F, look for "jpg" and you will know what to do, trust me), and let me know what more Darknet can deal with.

## Network configuration and stuff:
Once you labelled the images:
1) Put all the images and their .txt files with classes coordinates at Model/data/obj (replace the content already there);
2) Do the same with the classes.txt;
3) Insert images and .txt files at Model/data/test, they must be from the same nature of the ones at Model/data/obj (they can be even a subset from them, just be aware of the overfitting from it), to be used for the validation during the training process.
4) Run the generate_stuff.py with Python (and give me a tip for a better name for it, I haven't been really creative to do it);
5) Rename the desirable CFG file to model.cfg: if you gonna train a YOLOv4 file, rename the model_full.cfg to model.cfg, or if you gonna train a YOLOv4tiny file, rename the model_tiny.cfg to model.cfg;
6) Add a convolutional neural network's weight file to be used as a base on Model/backup named as model_last.weight, it can YOLOv4 or YOLOv4tiny (you can get a good YOLOv4 here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137, and also a good YOLOv4tiny here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29).

## Network training:
On the same folder where is the Model dir, install the Darknet (https://pjreddie.com/darknet/install/). Do it with CUDA resources if you can.

(Seriously, the training with no dedicated GPGPU may take **months** to reach a good couple of weights).

If you are like me and need to work five months only to buy a decent hardware (and still did not done it yet), you can use a good GPGPU from Google Colab for free (twelve hours per day at most, which usually is time enough to get a very nice couple of weights for your network), and if you live in USA or Canada, you can even rent processing power as an IaaS. You can access it at https://colab.research.google.com/; and you will need a Google account for it, of course.

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

#### (IT WILL WORK ONLY IF THE NOTEBOOK IS SET TO USE GPU)

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
So you trained the weights for a long time. Make it worth by accessing the backup directory and take the weight files (the Model_best.weights is the guy you have been waiting for).

Take the Model_best.weights, rename it to "Model.weights" and move it to the Networks directory (replace the one there). Also, move the Model.cfg and Model.names to the same directory.

You can do it onto a image file, a batch of image files or a video file, so, after that, take the video or image which you want to activate your brand new convolutional neural network, put at the root of project and rename it to "input.mp4" (if is an video sequence) or 'input.png" (if is an image). If it is a batch of images, just put them (JPG files) inside a directory named "input", and also create a directory named output.

And then sit down straight, correct your posture, take a deep breath, a sip of tea, and run the Activation_video.py, or Activation_image.py or Activation_images.py with Python, if for video sequence, a single image or a batch of images, respectively. And relax.

At this point, pay attention at the classes detection performance and not the FPS rate -- after all, you didn't trained your hardware but your convolutional neural network. At the end, a file called output.avi (if video sequence) or output.png (if an image) will be done and produced at the same directory, with all the video frames being executed at 20 FPS no matter how long your hardware take to process it (and feel free to hack the hyperparameters to change it). If you did it in a batch of images (with Activation_images.py), the outputs will all be insider the output directory, of course, with the same name of the original file inside the input directory (and no: the files aren't moved from a directory to another; the original file are all preserved).


### That's it. Regards and keep rocking, guys.
