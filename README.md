# StableVideo

## Installation
```
git clone https://github.com/rese1f/StableVideo.git
conda create -n stablevideo python=3.11
pip install -r requirements.txt
```

## Download Pretrained Model

All models and detectors can be downloaded from ControlNet Hugging Face page. [Download Link](https://huggingface.co/lllyasviel/ControlNet)


## Download sample videos


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
Run the following command to start. We provide some [prompt template](prompt_template.md) to help you achieve better result.
```
python app.py
```
the result `.mp4` video and keyframe will be stored in the directory `./log` after clicking `render` button.


## Acknowledgement

This implementation is built partly on [Text2LIVE](https://github.com/omerbt/Text2LIVE) and [ControlNet](https://github.com/lllyasviel/ControlNet).

<!-- ## Citation -->
