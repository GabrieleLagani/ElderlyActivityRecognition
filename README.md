
Experiments on human activity recognition from video based on deep learning.
This repository provides implementations of various deep learning models for 
video processing. In particular, we also provide the code for our 
Convolutional-Attentional 3D (CA3D) model (based on the CAST - 
Convolutional-Attentional Spatio Temporal block), and the Spatio-temporal 
Chi-stream Network (SChi-Net) model (based on the Chi-Stream block).
Code for training and evaluating the models on various datasets is available.
These are lightweight architectures, designed for computational efficiency,
and suitable for edge applications and constrained or consumer hardware.

## Update: new SchiNet architecture development in progress

The SchiNet architecture uses a novel tensor compression strategy that allows to 
reduce the memory and compute complexity to scale not as the product of dimensions, 
but as the sum. Moreover, additional compression strategies are explored to further 
improve the model footprint.

The work is still in progress, but preliminary results are available:

*Memory occupancy, GFLOPs, and processing frame rate measured over a 
forward-backward pass during training. Inputs: 64 frames of 112×112 pixels, 
mini-batches of size 32. Best results highlighted in **bold**.*

| Model | Memory (GB) ↓ | GFLOPs ↓ | Frames/s ↑ |
|-------|---------------|----------|------------|
| R3D-R18 | 28.0 | 32.4 | 90 |
| R2+1D-R18 | 44.8 | 29.6 | 42 |
| I3D | 13.2 | 17.2 | 367 |
| X3D-XL | 46.5 | 47.6 | 134 |
| SlowFast-R18 | 10.7 | 9.2 | 400 |
| MoViNet-A2 | 24.4 | 11.2 | 237 |
| STAM-B | 45.7 | 86.8 | 64 |
| TimesFormer-B | 40.6 | 67.6 | 66 |
| ViViT-2 | 18.9 | 25.2 | 205 |
| Swin3D-T | 12.5 | 15.0 | 373 |
| TubeViT-B | 18.3 | 36.8 | 139 |
| VideoMAE-v2 | 20.4 | 37.1 | 131 |
| | | | |
| **Ours (ST-XL)** | 11.5 | 23.8 | 236 |
| **Ours (SC-XL)** | **8.6** | **4.0** | **433** |
| **Ours (TC-XL)** | 10.4 | 11.3 | 416 |
| | | | |
| **Ours (ST-XL + quant.)** | 6.4 | 23.8 | 330 |
| **Ours (SC-XL + quant.)** | **4.7** | **4.0** | **606** |
| **Ours (TC-XL + quant.)** | 5.6 | 11.3 | 582 |


*Test accuracy of different methods on Kinetics400, considering both single-crop and multi-crop evaluation. Best results highlighted in **bold**.*

| Model | Acc. (%) ↑ (Single-Clip) | Acc. (%) ↑ (Single-Clip)¹ | Acc. (%) ↑ (Multi-Clip)² |
|-------|--------------------------|---------------------------|--------------------------|
| R3D-R50 | 52.25 | n/a³ | 71.80 |
| R2+1D-R50 | 53.20 | n/a³ | 72.00 |
| I3D | 51.67 | n/a³ | 71.10¹ |
| X3D-XL | 52.99 | n/a³ | 79.10 |
| SlowFast-R50 | 52.48 | n/a³ | 77.00 |
| MoViNet-A2 | 48.31 | n/a³ | 75.00¹ |
| STAM-B | 31.49 | n/a³ | 80.50¹ |
| TimesFormer-B | 32.35 | 48.62 | 78.00¹ |
| ViViT-2 | 32.79 | **66.16** | 84.30¹ |
| Swin3D-T | 40.00 | 56.16 | 77.80¹ |
| TubeViT-B | 34.33 | n/a³ | 88.60¹ |
| VideoMAE-v2 | n/a³ | 60.19 | 88.50¹ |
| | | | |
| **Ours (ST-XL + quant.)** | **53.59** | n/a³ | **88.63** |
| **Ours (SC-XL + quant.)** | 50.06 | n/a³ | 75.16 |
| **Ours (TC-XL + quant.)** | 51.49 | n/a³ | 80.26 |

---

**Footnotes:**
¹ Additional pre-training.  
² *10-LeftCenterRight* cropping evaluation.  
³ Not applicable/available.

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

## Acknowledgements
Research supported by project INAROS (INtelligenza ARtificiale per il 
mOnitoraggio e Supporto agli anziani), aimed at the development of deep 
learning technologies for video processing for elderly assistance in smart 
home and smart healthcare applications.

## Contacts
Gabriele Lagani: gabriele.lagani@isti.cnr.it
 
