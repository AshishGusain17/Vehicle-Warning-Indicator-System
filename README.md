
# Vehicle-Warning-Indicator-System
A deep learning and computer vision based warning system for vehicle drivers using live dash-cam footages.


```
Tracking of vehicles
The tracking of the vehicles with a track ID can be seen below.
```
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/tracking1.gif?raw=true" width="410">|
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/tracking2.gif?raw=true" width="410">

[Watch the complete video](https://youtu.be/LHlgFDLzG00)



<br />

```
Detection of the lanes. 
Whenever the driver gets out of the lane, he will be displayed a warning to stay inside the lane.
```
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/lanes1.gif?raw=true" width="410">|
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/lanes2.gif?raw=true" width="410">

[Watch the complete video](https://youtu.be/GF49Xrl0vCM)



<br />

```
Tail light detection
Detect all the tail lights of the vehicles applying brakes at night.
```
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/tail1.gif?raw=true" width="410">|
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/tail2.gif?raw=true" width="410">

[Watch the complete video](https://youtu.be/wOYojpGd03c)



<br />

```
Traffic signal recognition
Warning is shown when to stop and resume again using traffic lights.
```
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/traffic_signal1.gif?raw=true" width="410" height="235">|
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/traffic_signal2.gif?raw=true" width="410">



<br />
<br />

```
Vehicle collision estimation
Incase, a collision is estimated, driver is warned.
```
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/vehicle_collision1.gif?raw=true" width="410">|
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/vehicle_collision2.gif?raw=true" width="410">



<br />
<br />

```
Pedestrian stepping 
Whenever, pedestrian comes in our view, a warning is displayed.
```
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/pedestrian1.gif?raw=true" width="410">|
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/pedestrian2.gif?raw=true" width="410">



<br />

### Dependencies required:
* Python 3.0
* TensorFlow 2.0
* openCV

### Project Structure:
* **lanes**:This folder contains files related to lane detection only.
* **tf-color**: This folder contains files related to traffic light detection and detect the colour and accordingly give instructions to the driver.
* **tracked**: This folder contains detection and tracking algorithm for the vehicles.
* **untracked**: Detection and visualization only
* **utils**: contains various functions that are used continuously again and again for different frames.
* **estimations**: Detect pedestrians and vehicles too close to us that may cause collision.
* **cropping**: Cropping frames using drag and drop or clicking points.
* **display**: All the gifs shown above are stored here.

###  Requisities:

Download the  tensorflow model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
* Provide the path to the labels txt file using variable named PATH_TO_LABELS.
* Provide the path to the tensorflow model using variable named model_name.
* Make sure all the files are imported properly from the utils folder. If you get an error, add 
the location of the utils folder using sys module.
* Tensorflow version 2.0 is must or else you may come across various error.

### Working:
```
Run python integrate3.py or python intyolo.py after following the above mentioned requisities.
Now select the dash area for the car by clicking on multiple points as shown below. This is done to 
remove detection of our own vehicle in some cases which may generate false results.
```
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/crop1.png?raw=true" width="410">|

```
In the second step, select the area where searching of the lanes should be made. This may differ due to 
the placement of dash-cams in the vehicle. The area above the horizon where road ends should not be selected.
```
<img src="https://github.com/AshishGusain17/Vehicle-Warning-Indicator-System/blob/master/display/crop2.png?raw=true" width="410">|

```
Now, you can visualize thw working and see the warnings/suggestions displayed to the driver.
All the works that are implemented individually are present in their respective folders, which are integrated together.
Old models may have some bugs now, as many files inside utils are changed.
Visit honors branch of models repository forked from tf/models to see more work on this project, 
that I have done in google colab.
```

### Drawbacks:
* At night, searching for tail light should be made in the dark. If sufficient light is present, false cases can get introduced.
* Tracking works good for bigger objects, while smaller may loose their track ID at places.
* Threshold values used in lane detection needs to be altered depending on the roads and the quality of the videos.
* Object detection needs to work properly for better results throughout. The model with higher accuracy should be downloaded from the link given above.