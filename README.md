# **Traffic-Light-Detection**

#### Udacity Self Driving Car Engineer Nanodegree Program (Capstone Project)

## Project Introduction

The goal of this project is to train [Convolutional Neural Network (CNN)](https://de.wikipedia.org/wiki/Convolutional_Neural_Network) models for the Traffic Light Detection of the [Udacity Self-Driving Car Capstone Project](https://github.com/barrykidney/CarND-Capstone).
 
To detect traffic lights, [Carla](https://medium.com/udacity/how-the-udacity-self-driving-car-works-575365270a40) (Udacity’s actual Self-Driving Car) has a camera mounted behind the windshield, which periodically sends images to the Traffic Light Detector. The task of the Traffic Light Detector is to detect traffic lights and their respective color status in the pictures and to send the current color status to the planning module (Waypoint Updater), so that it stops the car at the stop line at 'red' or continues driving at 'green'.
The Traffic Light Detector uses a [Convolutional Neural Network (CNN)](https://de.wikipedia.org/wiki/Convolutional_Neural_Network) which has to be specially trained for the detection of traffic lights respectively their color states.

#### Scene from Simulator
![](assets/simulator_scene.png)

#### System Architecture
![](assets/System-Architecture.png)

For the Capstone project, it is necessary for the car to drive both a lap on the highway in the simulator and a lap on the course with Carla. While Carla provides 'real world' pictures of the environment, the simulator provides 'drawn' images, see example:

|Simulator (drawn image)         |Carla (real world image)   |
|--------------------------------|---------------------------|
|![](assets/simulator_sample.jpg)|![](assets/real_sample.jpg)|

For this reason, each own model for the simulator and for Carla must be trained.

## Prerequisites
To use this project the following programming environment is required:

* [CUDA 8.0](https://developer.nvidia.com/cuda-downloads)
* [cuDNN 6.0](https://developer.nvidia.com/cudnn)
* [TensorFlow 1.3](https://tensorflow.org)

NOTE: If you are using a different version of Tensorflow or CUDA than described above, your trained model will not be compatible with Carla!

Follow this [Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) to install the [Tensorflow Objecet-Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). 
But instead of using the original Tensorflow Object-Detection API, as described in the tutorial above, clone this [Traffic-Light-Detection](https://github.com/aboerzel/Traffic-Light-Detection) repository to a workspace directory `<project_dir>`.

## Traffic-Light-Detection
This project is based on the Tensorflow Object-Detection API and can be used to train models for the Traffic Light Detection and Classifier (Traffic Light Detection Node) of the Self-Driving Car Engineer Capstone Project.
The traffic light detection node takes a camera image as input, generates the bounding boxes around the detected traffic lights and supplies for each bounding box one score value for each of the 3 traffic light colors 'Red', 'Green', 'Yellow' or 'Unknown' if no assignment was possible.

The Object-Detection API offers the possibility to use several [Pre-Trained Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) as the basis for a fine tuning.
This project also makes it possible to train traffic light detection models very flexibly with different pre-trained models.
After a few tests, we decided to use the `ssd_inception_v2_coco_2017_11_17`, because it showed a good balance between performance and accuracy.

### Create Dataset:
The most important thing to train a practicable model is a data set corresponding to the respective problem!

This project uses the [Data Set](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view) from [Anthony Sarkis](https://medium.com/@anthony_sarkis), it includes a ready-labeled image collection for both the simulator and Carla.
But in the case that someone wants to use their own data set, the necessary steps are briefly described here.

##### 1. Collect Images
Collect images that correspond to the problem and store these in a sub-folder of the `data` folder.
Keep in mind to collect enough pictures per class and about the same number of pictures per class.    

##### 2. Label Images
In the next step, the objects to be detected (traffic lights) must be marked in the images (bounding box) and each provided with a class label (red, green, yellow).
This can be done with the tool [LabelImg](https://github.com/tzutalin/labelImg).
 
![](assets/LabelImg.PNG)

LabelImg saves a annotation file in the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) format for each labeled image in the `<annotations dir>`. 

##### 2. Create TFRecord Data Set
The following command can be used to create a [TFRecord](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564) data set from the images and the annotation files:  

```sh
(tf13) <project_directory>\object_detection> python create_pascal_tf_record.py 
    --label_map_path=data/ud_capstone_label_map.pbtxt
    --data_dir=data/<dataset dir> 
    --annotations_dir=data/<dataset dir>/<annotations dir>
    --output_dir=data/<dataset dir>/<dataset name>.TFRecord
```

The output is a single [TFRecord](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564) data set file, that can be used by the Object-Detection API to train a model.
	
### Train Model:
The following diagram shows the part of the project structure, which is important for the training of models.

The subdirectory `data` contains the two datasets for the simulator (sim_training_data) and for Carla (real_training_data). These contain the image files for the training as well as the corresponding [TFRecord](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564) data set file. Furthermore, here is the file `ud_capstone_label_map.pbtxt` located which maps the class IDs to label texts.

The [Pre-Trained Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), which are used as a basis for fine tuning, will be stored in the directory `pretrained_models`. Here, a corresponding subdirectory is created for each model type.

The checkpoints which are created during the training run will be stored in the `checkpoints` directory. These are stored in subdirectories named according to the model type.

The `fine_tuned_models` directory contains the exported models, which are ultimately used by the Traffic-Light-Detector. These are also stored in subdirectories named according to the model type.

The `config` directory contains the training pipeline configurations for the models to be trained. If further models are to be trained, new pipeline configurations must be added accordingly.

All other directories belong to the Object-Detection API and are not relevant for the training.

```
.<project_dir>\object_detection
├── data
│   ├── mixed_train.record
│   ├── mixed_eval.record
│   :
│       
├── models
│   └── ssd_inception_v2_coco_2018_01_28
│   │   ├── checkpoint
│   │   ├── frozen_inference_graph.pb
│   │   ├── model.ckpt.data-00000-of-00001
│   │   ├── model.ckpt.index
│   │   ├── model.ckpt.meta
│   │   └── saved_model
│   │       ├── saved_model.pb
│   │       └── variables
│   │
│   ├── ssd_mobilenet_v2_coco_2018_03_29
│   │   ├── ...
│   :   :
│
├── checkpoints
│   ├── ssd_inception_v2_coco_2018_01_28
│   │   ├── checkpoint
│   │   ├── model.ckpt-12345.*
│   │   :
│   │
│   ├── ssd_mobilenet_v2_coco_2018_03_29
│   │   ├── checkpoint
│   │   ├── model.ckpt-12345.*
│   │   :
│   :
│
├── fine_tuned_models
│   ├── ssd_inception_v2_coco_2018_01_28
│   │   ├── saved_model
│   │   ├── frozen_inference_graph.pb
│   │   :
│   │
│   ├── ssd_mobilenet_v2_coco_2018_03_29
│   │   ├── saved_model
│   │   ├── frozen_inference_graph.pb
│   │   :
│   :
│
└── config
   ├── labels_map.pbtxt
   ├── ssd_inception_v2.config
   ├── ssd_mobilenet_v2.config
   ├── ...
   :
```

To train an new model for the [Udacity Self-Driving Car Capstone Project](https://github.com/barrykidney/CarND-Capstone) follow these steps (Windows):

1. Open a Anaconda Prompt
```sh
(base) ..>
```

2. Activate your virtual environment, which you have set up for the Tensorflow Object-Detection API 1.3.
```sh
(base) ..> activate tf13
```

3. Set the environment variable `PROJECT_DIR` to the `<project_dir>` directory to which you have cloned this project:
```sh
(tf13) ..> set PROJECT_DIR=<project_directory>
```

4. Set the environment variable `PYTHONPATH` as follows:
```sh
(tf13) ..> set PYTHONPATH=%PROJECT_DIR%;%PROJECT_DIR%\slim;%PROJECT_DIR%\object_detection
```

5. Change to the object-detection directory
```sh
(tf13) ..> cd %PROJECT_DIR%\object_detection
```

6. Start the [Traffic Light Detection Tutorial](object_detection/traffic_light_detection_tutorial.ipynb) and follow the steps:
```sh
(tf13) <project_directory>\object_detection> jupyter notebook traffic_light_detection_tutorial.ipynb
```

7. Training Monitoring
To monitor the training, you can start [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) from a second console. Here you can see if and how well the model has already learned and whether the training process can be stopped.
```sh
(tf13) <project_directory>\object_detection> tensorboard --logdir=checkpoints/<model dir>
```
 
Tensorboard:
![](assets/tensorboard.png)

If you followed the steps of the tutorial above and trained a model, it will be located at:
```sh
<project_directory>\object_detection\fine_tuned_models\<dataset>_<model_type>\frozen_inference_graph.pb
```

To use this fine tuned model, the file must be copied to the ROS environment of the Capstone project and renamed accordingly.
The directory and the file names are exactly as expected by the traffic light recognition!
		
Simulator:
```sh
/ros/src/tl_detector/light_classification/ud_capstone_simulator_graph.pb
```

Site (Carla):
```sh
/ros/src/tl_detector/light_classification/ud_capstone_site_graph.pb
```
    	
### Test the Model
You can test the fine tuned model with the [Traffic Light Detection Tutorial](object_detection/traffic_light_detection_tutorial.ipynb) above, the results looks like this: 

|Carla (real world image)     |Simulator (drawn image)     |
|-----------------------------|----------------------------|
|![](assets/site-sample-1.png)|![](assets/sim-sample-1.png)|
|![](assets/site-sample-2.png)|![](assets/sim-sample-2.png)|
|![](assets/site-sample-3.png)|![](assets/sim-sample-3.png)|
|![](assets/site-sample-4.png)|![](assets/sim-sample-4.png)|
|![](assets/site-sample-5.png)|![](assets/sim-sample-5.png)|


Alternatively, a model can be tested with the TrafficLightClassifier.py, which is an equivalent implementation to the Traffic Light Detection Node implementation of the [Udacity Self-Driving Car Capstone Project](https://github.com/barrykidney/CarND-Capstone).
```sh
<project_directory>\python TrafficLightClassifier.py --frozen-model-path=./object_detection/fine_tuned_models/sim_ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb --image=./object_detection/data/sim_training_data/sim_data_capture/left0024.jpg
```

The result looks like this:
![](assets/TrafficLightClassifierSample.PNG)

