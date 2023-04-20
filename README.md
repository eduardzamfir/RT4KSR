# Towards Real-Time 4K Image Super-Resolution

**[Eduard Zamfir](https://scholar.google.com/citations?hl=en&user=5-FIWKoAAAAJ), [Marcos V. Conde](https://scholar.google.com/citations?user=NtB1kjYAAAAJ&hl=en), [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)**

[Computer Vision Lab, CAIDAS, University of Würzburg](https://www.informatik.uni-wuerzburg.de/computervision/home/)

Work part of the [NTIRE Real-Time 4K Super-Resolution](https://cvlai.net/ntire/2023/) Challenge @ CVPR 2023 in Vancouver

----
<img src="assets/rt4ksr_teaser.png" width="1000" />

## Abstract
Over the past few years, high-definition videos and images in 720p (HD), 1080p (FHD), and 4K (UHD) resolution have become standard. While higher resolutions offer improved visual quality for users, they pose a significant chal- lenge for super-resolution networks to achieve real-time performance on commercial GPUs. This paper presents a comprehensive analysis of super-resolution model designs and techniques aimed at efficiently upscaling images from 720p and 1080p resolutions to 4K. We begin with a simple, effective baseline architecture and gradually modify its design by focusing on extracting important high-frequency details efficiently. This allows us to subsequently downscale the resolution of deep feature maps, reducing the overall computational footprint, while maintaining high reconstruction fidelity. We enhance our method by incorporating pixel-unshuffling, a simplified and speed-up reinterpretation of the basic block proposed by NAFNet, along with structural re-parameterization. We assess the performance of the fastest version of our method in the new [NTIRE Real-Time 4K Super-Resolution](https://cvlai.net/ntire/2023/) challenge and demonstrate its potential in comparison with state-of-the-art efficient super-resolution models when scaled up. Our method was tested successfully on high-quality content from photography, digital art, and gaming content.

----

## Installation

- Create conda environment:
```
conda create --name rtsr python==3.10
source activate rt4ksr
```
- Install PyTorch (see [PyTorch instructions](https://pytorch.org/get-started/locally/)). For example,
```
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```
- Install the dependencies:
```
pip install -r requirements.txt
````

----
## Train

### Data Preparation
For faster I/O, we employ `data/preprocess/extract_subimages.py` to partition the training images into [512x512] segments. Our training dataset includes the complete sets of both [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and Flickr2K images. To compete in the [NTIRE Real-Time 4K Super-Resolution](https://cvlai.net/ntire/2023/) challenge, we add a subset of `1000` and `2500` images from [LSDIR](https://data.vision.ee.ethz.ch/yawli/index.html) (`shard-00`) and [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) (`part01`), respectively.

We generate bicubically downscaled LR images online and test our models on the standard benchmarks for Super-Resolution which can be found [here](https://cvnote.ddlee.cc/2019/09/22/image-super-resolution-datasets).

Define the dataset directory as follows:
````
dataroot/
|---DIV2K_Flickr2K/
|   |---sub/
|   |   |---train_HR/
|---GTA5
|   |---sub/
|---LSDIR
|   |---sub/
|---testsets/
|   |---set5
|   |   |---test/
|   |   |   |---HR/
|   |   |   |---f"LR_bicubic_x{scale}"/

...
````

### Define Model Architecture
You can modify the baseline architecture using following parser arguments:
- `arch`: `rtsrn`
- `type`: `base` | `rep`
- `num-blocks`: int
- `feature-channels`: int
- `act-type`: `gelu` | `lrelu` | `relu`
- `baseblock`: `base` | `dconvbase` | `nafbase` | `repbase` | `simplified`
- `downscaling`: bool
- `use-hfb`: bool
- `use-gaussian`: bool
- `unshuffle`: bool
- `layernorm`: bool
- `residual`: bool
- `squeeze`: bool
- `is-train`: bool
- `eca-gamma`: int

### Run Training
Train our baseline model for `x2` | `x3` Super-Resolution:
````
python main.py --arch rtsrn --type base --num-blocks 4 --feature-channels 24 --act-type gelu --baseblock base --scale [2|3] --crop-size 128 --batch-size 64 --dataset-name dif2k_sub --use-lsdir --lr-init 1e-4 --benchmark set5 set14 b100 urban100 
````

Train our RT4KSR model for `x2` | `x3` Super-Resolution:
````
python main.py --arch rtsrn --type rep --num-blocks 4 --feature-channels 24 --act-type gelu --baseblock simplified --scale [2|3] --crop-size 128 --batch-size 64 --dataset-name dif2k_sub --use-lsdir --lr-init 5e-4 --benchmark set5 set14 b100 urban100 --layernorm --is-train
````
With `--use-gta` and `--use-lsdir` you can add additional data to the model training. For enhanced performance, include the auxilary loss function using `--aux-loss` and `--loss-weight 1.0` to the above defined commands. 

After training is done, evaluation on the specified benchmarks will be performed automatically (only `set5`, `set14`, `b100` and `urban100` is supported). Make sure to follow previously described folder structure. In `experiments/` you can find bash scripts for running the baseline and RT4KSR model with default settings. 

### Test Model
You can find all the necessary details of testing the models in `test.py`. Make sure to correctly define the architecture.