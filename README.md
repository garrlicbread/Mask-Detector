# Mask-Detector

This is a real time facemask detector built on python using MobileNetV2, OpenCv & CAFFE Res10 Model.
Primarily, this tool predicts if a person is wearing a facemask or not.
Secondly, it can tell if the person is wearing it properly. If the person isn't wearing it properly, it prompts the person to adjust it.

PLEASE VIEW [OUTPUT.MD](Output.md) TO WATCH A GIF OF THE RESULT.

## Workings:
There are three main operations working together to make this project work and they consist of a neural-network, a pre-trained deep learning model and a function that combines the two together.

1) The first neural network is a simple mask detector. It was trained after using Google's MobileNetV2 as the base layer on a local machine in 20 minutes. 

2) The pre-trained deep learning model used is Berkley AI Research's (BAIR) Caffe Model whose weights were downloaded from their GitHub. This model detects a face in any given image. 

3) The final function connects the two models and uses OpenCv to display the results. Initially, it turns on the webcam and feeds every frame into an OpenCV method known as cv2.dnn.blobFromImage() which performs mean subtraction, scaling and channel swapping on the frames. Playing with these numbers determines how sensitive the model is to various colors found in every image/video. These frames are now fed into the Caffe Model that detects where the face is located in the image and returns the co-ordinates. These co-ordinates are fed into the mask detector which focuses on that area of the frame where a face is located and predicts if it detects a mask or not. OpenCV methods are used to draw rectangles to highlight the face and the color changes depending on the accuracy of the model. 

Known Issues:

1) The Model assumes spectacles to be a part of a mask. Assumed reason being lack of training images of people with glasses.
2) Due to lack of bearded people without masks in the dataset, the model sometimes cannot differentiate between a thick beard and a mask. Adding Gaussian Blur as a preprocessing step helped tackle this issue.

References:

1) Balaji Srinivasan: https://www.youtube.com/watch?v=Ax6P93r32KU [Mask Detector on Python using TF 1.15.2]
2) Murtaza's Workshop - Robotics and AI: https://www.youtube.com/watch?v=WQeoO7MI0Bs&t=9006s [Learn Open CV Basics]
3) Adrian Rosebrock: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/ [Face detection with OpenCV and deep learning]
4) Script to download models: https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py


The .png file in this repository is a graph depicting the training/validation accuracy vs loss. 
