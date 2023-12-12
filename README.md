
Experiments on human activity recognition from video for elderly monitoring.
Research supported by project INAROS (INtelligenza ARtificiale per il 
mOnitoraggio e Supporto agli anziani).

## Usage
Launch experiment with:
```
python exp.py --config <config> --device <device> --restart
```
Where:
 - `<config>` is the name of a configuration dictionary, with dotted 
 notation, defined anywhere in your code. For example
 `configs.base.config_base`.
 - `<device>` can be `cpu`, `cuda:0`, or any device you wish to use for
 the experiment.
 - The flag `--restart` is optional. If you remove it, you can resume a 
 previously suspended experiment from a checkpoint, if available.
 
 ## Datasets
[UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)
[HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database)

## Requirements
- Python  3.6
- PyTorch 1.8.1

# Contacts
Gabriele Lagani: gabriele.lagani@phd.unipi.it
 
