# Mask-Detector

This is a real time facemask detector built on python using MobileNetV2, OpenCv & CAFFE Res10 Model.
Primarily, this tool predicts if a person is wearing a facemask or not.
Secondly, it can tell if the person is wearing it properly. If the person isn't wearing it properly, it prompts the person to adjust it.

Notes:

1) The Model assumes spectacles to be a part of a mask. Assumed reason being lack of training images of people with glasses.
2) The Model is already trained so the saved model will be loaded. 
3) Execution time is < 10 seconds.
4) Due to lack of bearded people without masks in the dataset, the model sometimes cannot differentiate between a beard and a mask.
5) The CNN Architecture is commented out. Scroll down to view the code.
6) There is another section commented out which tests demo images and print if it detects a mask or not.

Links:

1) Res10 FP16 Model Weights Raw Link: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
2) Res10 FP16 Model Weights Git Link: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/weights.meta4
3) Res10 Model Prototxt Link: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/solver.prototxt
 
Note: A more convoluted version of this Res10 caffe model exists but no formal link could be found. They have to be trained manually for 8 hours. 

4) Raw Link to pre-trained model: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

References:
    
1) Balaji Srinivasan: https://www.youtube.com/watch?v=Ax6P93r32KU [Mask Detector on Python using TF 1.15.2]
2) Murtaza's Workshop - Robotics and AI: https://www.youtube.com/watch?v=WQeoO7MI0Bs&t=9006s [Learn Open CV Basics]
3) Adrian Rosebrock: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/ [Face detection with OpenCV and deep learning]

Here's what the output looks like: 

![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)
