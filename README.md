# CarND-Capstone Project

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

## Waypoint Updater

-Used udacity's straight approach

## Twist Controller

-Used udacity's straight approach

## TL Detector
### Neural Networks
 
 I created two networks for this project: a detector network and a classifier network. The detector network detects the traffic lights while the classifer network detects the state of the traffic light detected by the detector network.
 
Both of the programs for the models are stored in model_creator in the zip file in the corresponding detector and classifier folders. The saved models are stored in tl_detector/models in the zip file, which are named accordingly.

Models for the networks are state below:

** Detector **
UNet architecture: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

** Classifier **
InputLayer -> Conv2D -> MaxPooling2D -> Conv2D -> Flatten -> Dense -> Dense


The detector is trained with 50 epochs with a batch size of 16 on the simulator and on Carla.

The classifier is trained with 15 epochs and a batch size of 32 on the simulator. On Carla, the classifier uses the same epochs with a batch size of 64.

#### Data

The data in the detector is comprised of an image with either a green, yellow, red, or no traffic light with the corresponding traffic light masks.

The data in the classifier is comrpised of multiple images of a green, yellow, red, and no traffic light.

** Examples of Data **

 Detector:
 
 (carla)
 
 (simulator)
 
 Classifier:
 
 (carla)
 
 (simulator)
 
 
 
 ## Behavior
 
 The vehicle has a set behavior when to identify traffic lights. When the car is at a distance of 40 away from a traffic light, it starts identifying traffic lights. It then stops identifying traffic lights at a distance of 20. It does this because in a real world situation, if the light was green at the beginning of the intersection (distance of around 40) and turned red in the middle of the intersection (distance of around 20) the vehicle would not stop. 

(When vehicle is detecting traffic lights)

![vehicle_detecting](https://www.dropbox.com/s/xo1vl0t06tah7n1/diagram2.PNG?dl=0)

 (When vehicle stops detecting traffic lights)

![vehicle_not_detecting](https://www.dropbox.com/s/zogoklki324xe0p/diagram1.PNG?dl=0)

Also, I made the vehicle only detect traffic lights when it is going towards the traffic light to be more efficient. I did this by implementing a last distance variable to see if the vehicle is moving toward the traffic light or not by comparing it with the current distance.


