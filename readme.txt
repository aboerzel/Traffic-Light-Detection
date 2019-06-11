python model_main.py --pipeline_config_path=config/ssd_inception_v2.config --model_dir=checkpoints/ssd_inception_v2_coco_2018_01_28

tensorboard --logdir=checkpoints/ssd_inception_v2_coco_2018_01_28





python model_main.py --pipeline_config_path=config/ssd_mobilenet_v2.config --model_dir=checkpoints/ssd_mobilenet_v2_coco_2018_03_29

tensorboard --logdir=checkpoints/ssd_mobilenet_v2_coco_2018_03_29



prtoc.exe...

python Setup.py build
python Setup.py install

COCO API:
On Windows, run pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI


Activate virtual environent: activate tf-gpu

Training Anaconda Console:

    set PROJECT_DIR=D:\development\CarND\Traffic-Light-Detection
    set PYTHONPATH=%PROJECT_DIR%;%PROJECT_DIR%\slim;%PROJECT_DIR%\object_detection
    cd %PROJECT_DIR%\object_detection

    set MODEL=ssd_inception_v2_coco
    set DATASET=sim

    python train.py --logtostderr --pipeline_config_path=training/sim_ssd_inception_v2_coco.config --train_dir=checkpoints/sim_ssd_inception_v2_coco_2017_11_17/

    jupyter notebook traffic_light_detection_tutorial.ipynb

Monitoring with tensorboard:

 cd D:\development\CarND\Traffic-Light-Detection\object_detection
 tensorboard --logdir=checkpoints/sim_ssd_inception_v2_coco_2017_11_17

 tensorboard --logdir=checkpoints/real_ssd_inception_v2_coco_2018_01_28


Export graph:

cd D:\development\CarND\Traffic-Light-Detection\research\object_detection
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_inception_v2_coco.config --trained_checkpoint_prefix checkpoints/model.ckpt-1282 --output_directory fine_tuned_model/ssd_inception_v2_coco/
