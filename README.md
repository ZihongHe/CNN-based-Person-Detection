Video-Specific-Person-Detection based on algorithm similar to Region-CNN


1, how to run

change the image_load in cofiguration to the fold path that includes the specific person imgs
you would like to train(the more the better)

change the video_load in cofiguration to be the video path that you would like to detect.

type in terminal: python main.py


2, note

ensure configuration.csv are in the same path with main.py

it can only support mp4 format video

you could also modify configuration.csv to change video, video fps and any other variables

the program will automatically generate the model doc and the next time it will be automatically
loaded in.

it will generate a video (XXX_detection.mp4) of detecting the specific person with a red bounding box

3, configuration

usage of each configuration parameters could be viewed in the main sections in main.py

4, technique approach

Step 1:  read in specific-person image to be face imageset;  

Step 2:  read in block object from frames of video to be non-face imageset

Step 3:  build train-test set by random allocation

Step 4:  build Convolutional based Neural Network to recognise specific-person's face

step 5:  predict block object in each frame, and put red bounding box on the highest-prediction-score block frame

step 6:  the detection video will be generated








