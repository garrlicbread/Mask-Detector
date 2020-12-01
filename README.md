# Mask-Detector

This is a real time facemask detector built on python using MobileNetV2, OpenCv & CAFFE Res10 Model.
Primarily, this tool predicts if a person is wearing a facemask or not.
Secondly, it can tell if the person is wearing it properly. If the person isn't wearing it properly, it prompts the person to adjust it.

PLEASE VIEW OUTPUT.MD FOR THE RESULTS.

Known Issues:

1) The Model assumes spectacles to be a part of a mask. Assumed reason being lack of training images of people with glasses.
2) Due to lack of bearded people without masks in the dataset, the model sometimes cannot differentiate between a thick beard and a mask.

References:

1) Balaji Srinivasan: https://www.youtube.com/watch?v=Ax6P93r32KU [Mask Detector on Python using TF 1.15.2]
2) Murtaza's Workshop - Robotics and AI: https://www.youtube.com/watch?v=WQeoO7MI0Bs&t=9006s [Learn Open CV Basics]
3) Adrian Rosebrock: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/ [Face detection with OpenCV and deep learning]
4) Script to download models: https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py
