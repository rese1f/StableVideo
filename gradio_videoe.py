from share import *
import config

import os
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import imageio
from torchvision import transforms
from torchvision.transforms.functional import crop
import torch.nn.functional as F
from PIL import Image
import time

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from atlas.atlas_data import AtlasData
from atlas.atlas_utils import tensor2im


class VideoE:
    def __init__(self, base_cfg, canny_model_cfg, depth_model_cfg):
        self.base_cfg = base_cfg
        self.canny_model_cfg = canny_model_cfg
        self.depth_model_cfg = depth_model_cfg
        self.canny_model = None
        self.depth_model = None
        self.b_atlas = None
        self.f_atlas = None
        self.crops = None
    
    def load_canny_model(
        self,
        base_cfg='./models/cldm_v15.yaml',
        canny_model_cfg='./models/control_sd15_canny.pth',
    ):
        self.apply_canny = CannyDetector()
        canny_model = create_model(base_cfg).cpu()
        canny_model.load_state_dict(load_state_dict(canny_model_cfg, location='cuda'))
        canny_model = canny_model.cuda()
        self.canny_ddim_sampler = DDIMSampler(canny_model)
        self.canny_model = canny_model
        
    def load_depth_model(
        self,
        base_cfg='./models/cldm_v15.yaml',
        depth_model_cfg='./models/control_sd15_depth.pth',
    ):
        self.apply_midas = MidasDetector()
        depth_model = create_model(base_cfg).cpu()
        depth_model.load_state_dict(load_state_dict(depth_model_cfg, location='cuda'))
        depth_model = depth_model.cuda()
        self.depth_ddim_sampler = DDIMSampler(depth_model)
        self.depth_model = depth_model

    def load_video(self, video_name):
        self.data = AtlasData(video_name)
        # save video if not exist
        save_name = f"data/{video_name}/{video_name}.mp4"
        if not os.path.exists(save_name):
            imageio.mimwrite(save_name, self.data.original_video.cpu().permute(0, 2, 3, 1))
            print("original video saved.")
        toIMG = transforms.ToPILImage()
        self.f_atlas_origin = toIMG(self.data.cropped_foreground_atlas[0])
        self.b_atlas_origin = toIMG(self.data.background_grid_atlas[0])
        return save_name, self.f_atlas_origin, self.b_atlas_origin
    
    @torch.no_grad()
    def edit_foreground(self,
                        prompt="", 
                        a_prompt="best quality, extremely detailed", 
                        n_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",  
                        image_resolution=512, 
                        ddim_steps=20, 
                        scale=9, 
                        seed=-1, 
                        eta=0, 
                        low_threshold=100, 
                        high_threshold=200,
                        num_samples=1):

        if self.canny_model == None:
            self.load_canny_model(
                base_cfg=self.base_cfg,
                canny_model_cfg=self.canny_model_cfg
            )
        
        input_image = self.f_atlas_origin
        size = input_image.size
        model = self.canny_model
        ddim_sampler = self.canny_ddim_sampler
        apply_canny = self.apply_canny
        
        input_image = np.array(input_image)
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        # results = [255 - detected_map] + results
        self.f_atlas = Image.fromarray(results[0]).resize(size)
        return self.f_atlas
        
    @torch.no_grad()
    def edit_background(self,
                        prompt="", 
                        a_prompt="best quality, extremely detailed", 
                        n_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",  
                        image_resolution=512, 
                        detect_resolution=384,
                        ddim_steps=20, 
                        scale=9, 
                        seed=-1, 
                        eta=0,
                        num_samples=1):
                    
        if self.depth_model == None:
            self.load_depth_model(
                base_cfg=self.base_cfg,
                depth_model_cfg=self.depth_model_cfg
            )
        
        input_image = self.b_atlas_origin
        size = input_image.size
        model = self.depth_model
        ddim_sampler = self.depth_ddim_sampler
        apply_midas = self.apply_midas
        
        input_image = np.array(input_image)
        input_image = HWC3(input_image)
        detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(1)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        # results = [detected_map] + results
        self.b_atlas = Image.fromarray(results[0]).resize(size)
        return self.b_atlas
    
    @torch.no_grad()
    def render(self):
        # foreground
        if self.f_atlas == None:
            f_atlas = self.f_atlas_origin
        else:
            f_atlas = self.f_atlas
        f_atlas = transforms.ToTensor()(f_atlas).unsqueeze(0).cuda()
        f_atlas = torch.nn.functional.pad(
            f_atlas,
            pad=(
                self.data.foreground_atlas_bbox[1],
                self.data.foreground_grid_atlas.shape[-1] - (self.data.foreground_atlas_bbox[1] + self.data.foreground_atlas_bbox[3]),
                self.data.foreground_atlas_bbox[0],
                self.data.foreground_grid_atlas.shape[-2] - (self.data.foreground_atlas_bbox[0] + self.data.foreground_atlas_bbox[2]),
            ),
            mode="replicate",
        )
        foreground_edit = F.grid_sample(
            f_atlas, self.data.scaled_foreground_uvs, mode="bilinear", align_corners=self.data.config["align_corners"]
        ).clamp(min=0.0, max=1.0)
        
        foreground_edit = foreground_edit.squeeze().t()  # shape (batch, 3)
        foreground_edit = (
            foreground_edit.reshape(self.data.config["maximum_number_of_frames"], self.data.config["resy"], self.data.config["resx"], 3)
            .permute(0, 3, 1, 2)
            .clamp(min=0.0, max=1.0)
        )
        # background
        if self.b_atlas == None:
            b_atlas = self.b_atlas_origin
        else:
            b_atlas = self.b_atlas
        b_atlas = transforms.ToTensor()(b_atlas).unsqueeze(0).cuda()
        background_edit = F.grid_sample(
            b_atlas, self.data.scaled_background_uvs, mode="bilinear", align_corners=self.data.config["align_corners"]
        ).clamp(min=0.0, max=1.0)
        background_edit = background_edit.squeeze().t()  # shape (batch, 3)
        background_edit = (
            background_edit.reshape(self.data.config["maximum_number_of_frames"], self.data.config["resy"], self.data.config["resx"], 3)
            .permute(0, 3, 1, 2)
            .clamp(min=0.0, max=1.0)
        )
        
        output_video = (
                self.data.all_alpha * foreground_edit
                + (1 - self.data.all_alpha) * background_edit
        )
        if not os.path.exists('log'):
            os.mkdir('log')
        save_name = f"log/{time.time()}.mp4"
        imageio.mimwrite(save_name, output_video.cpu().permute(0, 2, 3, 1))
        return save_name
    
    @torch.no_grad()
    def advanced_edit(self):
        if self.crops == None:
            self.crops = self.data.get_global_crops_multi()
        return


if __name__ == '__main__':
    videoe = VideoE(base_cfg="/home/syh/ControlNet/models/cldm_v15.yaml",
                    canny_model_cfg="/home/syh/ControlNet/models/control_sd15_canny.pth",
                    depth_model_cfg="/home/syh/ControlNet/models/control_sd15_depth.pth")
    
    f = lambda x, y: transforms.ToPILImage()(x).save(f'{y}.png') # debug
    videoe.load_video("car-turn")
    videoe.advanced_edit()
    
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## VideoE")
        with gr.Row():
            with gr.Column():
                original_video = gr.Video(label="Original Video", interactive=False)
                with gr.Row():
                    foreground_atlas = gr.Image(label="Foreground Atlas", type="pil")
                    background_atlas = gr.Image(label="Background Atlas", type="pil")
                gr.Markdown("### Step 1. select one example video and click **Load Video** buttom and wait for 10 sec.")
                avail_video = [f.name for f in os.scandir("./data") if f.is_dir()]
                video_name = gr.Radio(choices=avail_video,
                                      label="Select Example Videos")
                load_video_button = gr.Button("Load Video")
                gr.Markdown("### Step 2. write text prompt and advanced options for background and foreground.")
                with gr.Row():
                    f_prompt = gr.Textbox(label="Foreground Prompt")
                    b_prompt = gr.Textbox(label="Background Prompt", value="winter scene, snowy scene, beautiful snow")
                with gr.Row():
                    with gr.Accordion("Advanced Foreground Options", open=False):
                        f_image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=256)
                        f_low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                        f_high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                        f_ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                        f_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=15.0, value=9.0, step=0.1)
                        f_seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        f_eta = gr.Number(label="eta (DDIM)", value=0.0)
                        f_a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                        f_n_prompt = gr.Textbox(label="Negative Prompt",
                                            value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
                    with gr.Accordion("Advanced Background Options", open=False):
                        b_image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=256)
                        b_detect_resolution = gr.Slider(label="Detect Resolution", minimum=256, maximum=768, value=384, step=128)
                        b_ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                        b_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=15.0, value=9.0, step=0.1)
                        b_seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        b_eta = gr.Number(label="eta (DDIM)", value=0.0)
                        b_a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                        b_n_prompt = gr.Textbox(label="Negative Prompt",
                                            value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
                gr.Markdown("### Step 3. edit each one and render.")
                with gr.Row():
                    f_run_button = gr.Button("Edit Foreground")
                    b_run_button = gr.Button("Edit Background")
                run_button = gr.Button("Render")
            with gr.Column():
                output_video = gr.Video(label="Output Video", interactive=False)
                with gr.Row():
                    output_foreground_atlas = gr.Image(label="Output Foreground Atlas", type="pil", interactive=False)
                    output_background_atlas = gr.Image(label="Output Background Atlas", type="pil", interactive=False)
        
        # edit param
        f_edit_param = [f_prompt, f_a_prompt, f_n_prompt, f_image_resolution, f_ddim_steps, f_scale, f_seed, f_eta, f_low_threshold, f_high_threshold]      
        b_edit_param = [b_prompt, b_a_prompt, b_n_prompt, b_image_resolution, b_detect_resolution, b_ddim_steps, b_scale, b_seed, b_eta]
        # action
        load_video_button.click(fn=videoe.load_video, inputs=video_name, outputs=[original_video, foreground_atlas, background_atlas])
        f_run_button.click(fn=videoe.edit_foreground, inputs=f_edit_param, outputs=[output_foreground_atlas])
        b_run_button.click(fn=videoe.edit_background, inputs=b_edit_param, outputs=[output_background_atlas])
        run_button.click(fn=videoe.render,outputs=[output_video])
        
    # block.launch(server_name='0.0.0.0')
    block.launch(share=True)
