
# RADNET

Rock Art Detection Network.

Built for usage with Keras and Tensorflow backend.

Code repository that has been based on:

* https://github.com/kbardool/keras-frcnn
* https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras

### Data

Create a data folder through using the preprocess tool in the sibling folder.

### Files

- **train.py** - Training a model from scratch.
- **cont_train.py** - Continue to train from an existing model.
- **test.py** - Evaluate model after training.
- **predict.py** - Make predictions from a trained model on a single panel.
- **test_data.py** - Script for exploring training data and see that data processing is working properly.
- **faster_rcnn/**
	- **augmentation.py** - Methods for performing data augmentation.
	- **config.py** - Defines model configuration.
	- **losses.py** - Includes all losses used when training.
	- **RADNet.py** - Class for creating the network for prediction.
	- **RoiPoolingConv.py** - Class for performing RoI Pooling.
	- **rpn.py** - Methods for creating the RPN network together with helper functions. 
	- **utils.py** - Various utility functions for reading and feeding data.
	- **base_models/** - Different base networks for processing images.

### Setup

1. Create model folder in root directory.
2. Create environment based on **environment.yml**.
3. Download the model weights [here](https://www.dropbox.com/s/sbgdx4zy0w8dlp5/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5?dl=1)
4. Place the weights in **faster_rcnn/base_models/**

### Run

1. Setup **faster_rcnn/config.py**.
2. Run **test_data.py** to see that data reading/feeding works and get some plots and outputs that could be used for changing the config.
3. Edit **test.py** and **cont_train.py** to setup epoch length, nr of epochs and if validation is going to be applied.
4. For running training and testing in background: ```nohup bash -c 'python train.py; sleep 60; python cont_train.py; sleep 60; python test.py' &```
