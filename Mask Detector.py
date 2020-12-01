"""
@author: Sukant Sindhwani

Real Time Mask Detector using MobileNetV2, OpenCv & CAFFE Res10 Model.
Primarily, this tool predicts if a person is wearing a facemask or not.
Secondly, it can tell if the person is wearing it properly.
If the person isn't wearing it properly, it prompts the person to adjust it.

Links:

1) Res10 FP16 Model Weights Raw Link: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
2) Res10 FP16 Model Weights Git Link: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/weights.meta4
3) Res10 Model Prototxt Link: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/solver.prototxt
 
Note: A more convoluted version of this Res10 caffe model exists but no formal link could be found. They have to be trained manually for 8 hours. 

4) Raw Link to pre-trained model: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

"""

# Initializing starting time
import time
program_execution_start = time.time()

# Importing Libraries 
import os
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, Input

# Loading the training/test loss graph
load_img(r"C:\\Desktop\Projects\Mask Detector\Accuracy & Loss.png")

# Loading the pre-trained Mask Detection model
mask_detector_path = r"C\\Desktop\Python\Projects\Mask Detector\Saved Weights\Mask Detector.model"
mask_detector_model = load_model(mask_detector_path)

# Loading the pre-trained Caffe Model that detects faces
prototxt_path = "C://Desktop/Python/Projects/Mask Detector/Face Detecting Pre-Trained Models/CAFFE Models/Res10/deploy.prototxt"
caffe_model_path = "C://Desktop/Python/Projects/Mask Detector/Face Detecting Pre-Trained Models/CAFFE Models/Res10/res10_300x300_ssd_iter_140000.caffemodel"
caffe_model = cv2.dnn.readNetFromCaffe(prototxt_path, caffe_model_path) 

# Defining a function that detects a face and then predicts if that face has a mask
def face_and_mask_detection(video, facedetector, maskdetector):
    (h, w) = video.shape[:2]
    # avg = np.array(video).mean(axis=(0, 1)) 
    # Constructing input blob for the frames by resizing to a fixed 300 * 300 pixels and normalizing it.
    # Try removing the cv2.resize if any issues arise
    blob = cv2.dnn.blobFromImage(cv2.resize(video, (300, 300)), 1.0, (224, 224), (100.0, 115.0, 125.0))        
    #  Other  color average combinations to try: A) (100.0, 115.0, 125.0)  B) (100.0, 100.0, 100.0) C) (123.0, 135.0, 105.0)
    
    # Setting the blob as input 
    facedetector.setInput(blob) 
    detections = facedetector.forward()
    
    # Initializing lists
    faces = [] 
    locations = []
    predictions = [] # Of the mask detector model

    # Looping over the detections
    for i in range(0, detections.shape[2]):
        
        # Extract the confidence/probability of detecting a face 
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by ensuring confidence is greater than threshold
        if confidence > 0.7:
            # Compute the (x, y) co-ords of bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
       
            # Pre-processing detected faces to predict mask probability
            face = video[startY:endY, startX:endX]
            face = cv2.GaussianBlur(face, (5, 5), 0)  # Adding the blur improved performance on faces with beards. See Known issues in README.md to know more.
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            # Adding the faces and boxes to their lists
            faces.append(face)
            locations.append((startX, startY, endX, endY))
            
        # Making sure at least one face is detected before making predictions
        if len(faces) > 0:
            faces = np.array(faces, dtype = 'float32')
            predictions = maskdetector.predict(faces, batch_size = 32)
            predictions = list(predictions)

        return (locations, predictions)
    
# Initializing ending time 
program_execution_end = time.time()

# Calculating total time taken for execution
total_time = (program_execution_end - program_execution_start)
print()
print(f"This model took {round(total_time, 2)} seconds to load.")
  
## Initializing the cameras
        
# # For when using cv2.VideoCapture()
# videocam = cv2.VideoCapture(0)
# videocam.set(3, 1280)
# videocam.set(4, 720)

# For when using imutils.video.VideoStream()
videocam = VideoStream(src = 0, framerate = 32).start()

# Looping over the frames from the Video stream
while True:
    frames = videocam.read() # For when using imutils.video.Videostream
    # _, frames = videocam.read() # For when using cv2.VideoCam(0)
    frames = imutils.resize(frames, width = 1000, height = 1000)
    frames = cv2.flip(frames, 1)

    # Detect the faces and masks 
    (location, prediction) = face_and_mask_detection(frames, caffe_model, mask_detector_model)
  
    # Loop over the detected faces. For this, we'll use the zip function
    # It takes in iterables as arguments and returns an iterator. 
    # This iterator generates a series of tuples containing elements from each iterable.
    for (rect, predictions) in zip(location, prediction):   
        (startX, startY, endX, endY) = rect # Storing all four points representing a face in the rect variable
        (mask, nomask) = predictions
        mask = round(mask, 2)
        nomask = round(nomask, 2)
        
        # text = ("Mask:", mask, "No Mask:", nomask) # To display only the predictions
        
        if mask >= 0.90:
            text = "Mask worn correctly. Model Surety: {:.2f}".format(max(mask, nomask) * 100)
            color = (204, 209, 72)                
        elif nomask >= 0.70:
            text = "No mask detected. Model Surety: {:.2f}".format(max(mask, nomask) * 100)
            color = (0, 69, 255)                 
        elif mask < 0.90 and nomask < 0.70:
            text = "Please adjust your mask. Model Surety: {:.2f}".format(mask * 100)
            color = (0, 255, 255)
  
        y = startY - 10 # To put the text above the rectangle
        cv2.putText(frames, str(text), (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1)
        cv2.rectangle(frames, (startX, startY), (endX, endY), color, 1)
        
    cv2.imshow("Camera:", frames)
    
   	# if the Escape key is pressed; break
    if cv2.waitKey(1) % 256 == 27:
        break

# Ending the loop
videocam.stream.release() # For when using imutils.video.VideoStream
# videocam.release() # For when using cv2.VideoCapture
cv2.destroyAllWindows()
     
#################################### Training the CNN ######################################################################################################

# # Initializing hyperparameters and directories
# # Default values commented out
# epochs = 30                                      
# learning_rate = 0.0001                          
# batch_size = 32                                
# df_path = "D:\Datasets\Mask Dataset"
# classes = ['With Mask', 'Without Mask']

# print()
# print("Pre-processing the data...")

# # Creating two lists and appending the images & labels to those lists
# df = []
# class_labels = []

# for i in classes:
#     path = os.path.join(df_path, i)
#     for images in os.listdir(path):
#         img_path = os.path.join(path, images)
#         image = load_img(img_path, target_size = (224, 224))
#         image = img_to_array(image)
#         image = preprocess_input(image)
#         df.append(image)
#         class_labels.append(i)
        
# # Perform Label Binarizer encoding 
# label_binarizer = LabelBinarizer()
# class_labels = label_binarizer.fit_transform(class_labels)
# class_labels = to_categorical(class_labels)
        
# # Convert the data and labels to arrays of float32
# df = np.array(df, dtype = 'float32')
# class_labels = np.array(class_labels)

# # Splitting df into training and test set 
# X_train, X_test, y_train, y_test = train_test_split(df, class_labels, test_size = 0.2, stratify = class_labels)

# Training the model

# # Data augmentation using ImageDataGenerator 
# df_aug = ImageDataGenerator(rotation_range = 40,
#                             shear_range = 0.2,
#                             zoom_range = 0.2,
#                             height_shift_range = 0.2,
#                             width_shift_range = 0.2,
#                             horizontal_flip = True,
#                             fill_mode = "nearest")

# # Loading Mobilenet model as the Base model (Head layers to be left off)
# base_model = MobileNetV2(include_top = False, weights = 'imagenet',
#                          input_tensor = Input(shape = (224, 224, 3)))        

# # Head model that will be placed on top of base model 
# head_model = base_model.output
# head_model = AveragePooling2D(pool_size = (7, 7))(head_model)          # Can try MaxPooling2D as well
# head_model = Flatten(name = "Flatten")(head_model)                     
# head_model = Dropout(0.3)(head_model)                                  # Default dropout = 0.5 
# head_model = Dense(units = 2, activation = "softmax")(head_model)      # Can try "Sigmoid" as well

# model = Model(inputs = base_model.inputs, outputs = head_model)

# # Freezing all layers in Base Model so they don't train when we begin the training
# # We don't want to train the base model because it's already trained!
# for layer in base_model.layers:
#     layer.trainable = False
    
# # Compiling the Model
# model.compile(optimizer = Adam(lr = learning_rate,                      # Can try Adamax as well
#                                decay = learning_rate / epochs),
#               loss = 'binary_crossentropy', 
#               metrics = ['accuracy', 'mse'])

# # Training 
# print()
# print("Training the model...")
# model_fitted = model.fit(df_aug.flow(X_train, y_train, batch_size = batch_size),
#                          epochs = epochs,
#                          verbose = 1,
#                          validation_data = (X_test, y_test),
#                          steps_per_epoch = len(X_train) // batch_size,  # Should ideally be TotalTrainingSamples / TrainingBatchSize
#                          validation_steps = len(X_test) // batch_size)  # Should ideally be TotalvalidationSamples / ValidationBatchSize

# # Saving the model 
# model.save("C:/Users/Sukant Sidnhwani/Desktop/Python/Projects/Mask Detector/Saved Weights/Mask Detector.model", save_format = 'h5')

# # Predicting the test set 
# predictions = model.predict(X_test, batch_size = batch_size)
    
# # Finding labels of predicted test set results 
# predictions = np.argmax(predictions, axis = 1)

# # Creating a classification report card 
# print() 
# print("The model has been trained. Printing report card now.")
# print(classification_report(y_true = y_test.argmax(axis = 1),
#                             y_pred = predictions, 
#                             target_names = label_binarizer.classes_))   
# print()

########################################################################################################################################################333

# # Plotting the training loss and accuracy

# plt.style.use("dark_background")
# plt.figure(figsize=(12, 8))
# plt.plot(np.arange(0, epochs), model_fitted.history['loss'], label = "Training Loss", color = "red")
# plt.plot(np.arange(0, epochs), model_fitted.history['val_loss'], label = "Testing Loss", color = "red")
# plt.plot(np.arange(0, epochs), model_fitted.history['accuracy'], label = "Training Accuracy", color = "cyan")
# plt.plot(np.arange(0, epochs), model_fitted.history['val_accuracy'], label = "Testing Accuracy", color = "cyan")
# plt.title("Training/Testing Loss and Accuracies")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy/Loss")
# plt.legend(loc = 'center right')
 
################################################ Predicting a single demo image #######################################################

# The following function and loop is for testing demo images

# def demo_pred(number):
#     demo = load_img(f"C://Desktop/Python/Projects/Mask Detector/Demo Images/Demo{number}.jpg",                      # Change the path file to your demo images folder
#                     target_size = (224, 224))
#     demo = img_to_array(demo)
#     demo = preprocess_input(demo)
#     demo = np.array(demo, dtype = 'float32')
#     demo = np.expand_dims(demo, axis = 0)
#     demo_prediction = model.predict(demo, batch_size = 32)
#     demo_percent = np.amax(demo_prediction)
#     demo_percent = round(demo_percent * 100, 2)
#     demo_prediction = np.argmax(demo_prediction, axis = 1)
#     demo_prediction = "IS wearing a mask." if demo_prediction == [0] else "IS NOT wearing a mask."
#     print("The person in this demo image", demo_prediction, f"Model Surety : {demo_percent}%")

# while True:
#     try:
#         num = input("Enter the demo image number you want to test: ")
#         demo_pred(num)
#     except:
#         print("Image not found.")
#         break
    
# ##########################################################################################################################################
