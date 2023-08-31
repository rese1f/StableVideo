# StableVideo

[![](http://img.shields.io/badge/cs.CV-arXiv%3A2308.09592-B31B1B.svg)](https://arxiv.org/abs/2308.09592)
[![](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-orange)](https://huggingface.co/spaces/Reself/StableVideo)

> **StableVideo: Text-driven Consistency-aware Diffusion Video Editing**  
> Wenhao Chai, Xun Guo✉️, Gaoang Wang, Yan Lu  
> _ICCV 2023_

https://github.com/rese1f/StableVideo/assets/58205475/558555f1-711c-46f0-85bc-9c229ff1f511

https://github.com/rese1f/StableVideo/assets/58205475/c152d0fa-16d3-4528-b9c2-ad2ec53944b9

https://github.com/rese1f/StableVideo/assets/58205475/0edbefdd-9b5f-4868-842c-9bf3156a54d3

## Colab Notebook Example
https://colab.research.google.com/drive/1NkOw70r1ObVT-Y9Do2e7jQF6geoiYTZi?usp=sharing

## VRAM requirement
|   |VRAM (MiB)|
|---|---|
|float32|29145|
|amp|23005|
|amp + cpu|17639|
|amp + cpu + xformers|14185|

- cpu: use cpu cache, args: `save_memory`

under default setting (*e.g.* resolution, *etc.*) in `app.py`

## Installation
```
git clone https://github.com/rese1f/StableVideo.git
conda create -n stablevideo python=3.11
pip install -r requirements.txt
(optional) pip install xformers 
```

(optional) We also provide CPU only version [huggingface demo](https://huggingface.co/spaces/Reself/StableVideo).
```
git lfs install
git clone https://huggingface.co/spaces/Reself/StableVideo
pip install -r requirements.txt
```

## Download Pretrained Model

All models and detectors can be downloaded from ControlNet Hugging Face page at [Download Link](https://huggingface.co/lllyasviel/ControlNet).


## Download example videos
Download the example atlas for car-turn, boat, libby, blackswa, bear, bicycle_tali, giraffe, kite-surf, lucia and motorbike at [Download Link](https://www.dropbox.com/s/oiyhbiqdws2p6r1/nla_share.zip?dl=0) shared by [Text2LIVE](https://github.com/omerbt/Text2LIVE) authors.

You can also train on your own video following [NLA](https://github.com/ykasten/layered-neural-atlases).

And it will create a folder data:
```
StableVideo
├── ...
├── ckpt
│   ├── cldm_v15.yaml
|   ├── dpt_hybrid-midas-501f0c75.pt
│   ├── control_sd15_canny.pth
│   └── control_sd15_depth.pth
├── data
│   └── car-turn
│       ├── checkpoint # NLA models are stored here
│       ├── car-turn # contains video frames
│       ├── ...
│   ├── blackswan
│   ├── ...
└── ...
```

## Run and Play!
Run the following command to start.
```
python app.py
```
the result `.mp4` video and keyframe will be stored in the directory `./log` after clicking `render` button.

You can also edit the mask region for the foreground atlas as follows. Currently there might be a bug in Gradio. Please carefully check if the `editable output foreground atlas block` looks the same as the one above. If not, try to restart the entire program.

<img width="916" alt="" src="https://github.com/rese1f/StableVideo/assets/58205475/ec8dd9f0-84fb-43ca-baaa-fb6c58da0d77">


## Citation
If our work is useful for your research, please consider citing as below. Many thanks :)
```
@article{chai2023stablevideo,
  title={StableVideo: Text-driven Consistency-aware Diffusion Video Editing},
  author={Chai, Wenhao and Guo, Xun and Wang, Gaoang and Lu, Yan},
  journal={arXiv preprint arXiv:2308.09592},
  year={2023}
}
```

## Acknowledgement

This implementation is built partly on [Text2LIVE](https://github.com/omerbt/Text2LIVE) and [ControlNet](https://github.com/lllyasviel/ControlNet).

<!-- ## Citation -->
