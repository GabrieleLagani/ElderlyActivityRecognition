
Experiments on human activity recognition from video based on deep learning.
Research supported by project INAROS (INtelligenza ARtificiale per il 
mOnitoraggio e Supporto agli anziani), aimed at the development of deep 
learning technologies for video processing for elderly assistance in smart 
home and smart healthcare applications.

## Models
This repository provides implementations of various deep learning models for 
video processing. In particular, we also provide the code for our 
Convolutional-Attentional 3D (CA3D) model (based on the CAST - 
Convolutional-Attentional Spatio Temporal block).
Code for training and evaluating the models on various datasets is available.

## Usage
Launch experiment with:
```
python runexp.py --config <config> --mode <train|test|traintest> --device <device> --restart
```
Where:
 - `<config>` is the name of a configuration dictionary, with dotted 
 notation, defined anywhere in your code. For example
 `configs.base.config_base`.
  - `<mode>` can be one of `train`, `test`, `traintest`, depending if you 
  want to perform model training, testing, or both.
 - `<device>` can be `cpu`, `cuda:0`, or any device you wish to use for
 the experiment.
 - The flag `--restart` is optional. If you remove it, you can resume a 
 previously suspended experiment from a checkpoint, if available.
 
 ## Datasets
[UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)

[HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database)

[Kinetics](https://github.com/cvdfoundation/kinetics-dataset)


## Requirements
- Python  3.10
- PyTorch 2.0.1

## Contacts
Gabriele Lagani: gabriele.lagani@phd.unipi.it
 
