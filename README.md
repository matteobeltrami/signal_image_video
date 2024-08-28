## Signal, Image and Video project

In this notebook we analyze the impact of different types of image preprocessing on the performance of an object detection neural network.
In particular, for this analysis, a network incorporated in the [micromind](https://github.com/micromind-toolkit/micromind) toolkit and developed within the [E3DA](https://e3da.fbk.eu/) laboratory of the Bruno Kessler foundation was used as a detection model.
The peculiarity of this object detection model is that it requires a reduced computational resource, while maintaining robust performance.

In order to reproduce the results and the analysis you can follow some simple steps.

If you just want to read and understand the analysis you can simply run the [`analysis.ipynb`](https://github.com/matteobeltrami/signal_image_video/blob/train_optim/analysis.ipynb) notebook using the pre-trained models available in the `models` folder.

If instead you want to train and evaluate the models on your own, brace yourself and follow these simple steps: 


### Setup the environment
First of all create and activate your python environment:
```
conda create -n siv python=3.9
conda activate siv
```

Clone the repository:
```
git clone https://github.com/matteobeltrami/signal_image_video.git
```

Open micromind and install it in editable mode:
```
cd signal_image_video/micromind
pip install -e . 
```

Install the extra requirements for the object detection recipe:
```
pip install -r extra_requirements
```

### Train your network
Inside `recipes/object_detection` launch
```
python train.py cfg/yolo_phinet.py --experiment_name=<EXPERIMENT>
```
making sure to set the right arguments (most importantly the `input_shape` one) inside the `cfg/yolo_phinet.py` configuration file and the right preprocessing inside the `train.py` script.

### Evaluate your network
Inside `recipes/object_detection` launch
```
python validate.py cfg/yolo_phinet.py --experiment_name=<EXPERIMENT>
```
making sure to set the right arguments (most importantly the `input_shape` one) inside the `cfg/yolo_phinet.py` configuration file and the right preprocessing inside the `validate.py` script.

#### Last but not least
Have fun playing with all the potential this toolkit has to offer. 
Read the [micromind.md](https://github.com/matteobeltrami/signal_image_video/blob/train_optim/micromind.md) presentation file and the [documentation](https://micromind-toolkit.github.io/docs/) for more info on micromind.

## Contacts
Project developed by Matteo Beltrami.
- email: beltramimatteo01@gmail.com
- github: [@matteobeltrami](https://github.com/matteobeltrami)