from share import *
import config

import os
import shutil  
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import torch.optim as optim
import random
import imageio
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import time
import scipy.interpolate
from tqdm import tqdm

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from atlas.atlas_data import AtlasData
from atlas.atlas_utils import get_grid_indices, get_atlas_bounding_box


class VideoE:
    def __init__(self, base_cfg, canny_model_cfg, depth_model_cfg):
        self.base_cfg = base_cfg
        self.canny_model_cfg = canny_model_cfg
        self.depth_model_cfg = depth_model_cfg
        self.img2img_model = None
        self.canny_model = None
        self.depth_model = None
        self.b_atlas = None
        self.f_atlas = None
        self.data = None
        self.crops = None
    
    def load_img2img_model(self):
        from diffusers import StableDiffusionImg2ImgPipeline

        # load the pipeline
        device = "cuda"
        self.img2img_model = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
        ).to(device)
    
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
    def canny_edit(self, input_image=None,
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
    def depth_edit(self, input_image=None,
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
    def edit_foreground(self, *args, **kwargs):
        input_image = self.f_atlas_origin
        self.canny_edit(input_image, *args, **kwargs)
        return self.f_atlas
    
    @torch.no_grad()
    def edit_background(self, *args, **kwargs):
        input_image = self.b_atlas_origin
        self.depth_edit(input_image, *args, **kwargs)
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
        id = time.time()
        os.mkdir(f"log/{id}")
        save_name = f"log/{id}/video.mp4"
        imageio.mimwrite(save_name, (255 * output_video.detach().cpu()).to(torch.uint8).permute(0, 2, 3, 1))
        
        keyframes = range(70)
        for index in keyframes:
            try:
                transforms.ToPILImage()(output_video[index]).save(f"log/{id}/{index}.png")
            except:
                continue
        return save_name
    
    def advanced_edit_foreground(self, 
                                 keyframes, 
                                 res=2000,
                                 *args, **kwargs):
        keyframes = [int(x) for x in keyframes.split(",")]
        
        if self.data is None:
            raise ValueError("Please load video first")
        self.crops = self.data.get_global_crops_multi(keyframes, res)
        n_keyframes = len(keyframes)
        indices = get_grid_indices(0, 0, res, res)
        f_atlas = torch.zeros(size=(n_keyframes, res, res, 3,)).to("cuda")

        img_list = [transforms.ToPILImage()(i[0]) for i in self.crops['original_foreground_crops']]
        # Get size of images
        widths, heights = zip(*(i.size for i in img_list))
        # Concatenate images horizontally
        merged_image = Image.new('RGB', (sum(widths), max(heights)))
        x_offset = 0
        for img in img_list:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        # Perform some operation on merged image
        with torch.no_grad():
            merged_image = self.canny_edit(merged_image, *args, **kwargs)
        f_img = []
        
        # Split the merged image back into individual images
        x_offset = 0
        for frame, img in enumerate(img_list):
            f_value = merged_image.crop((x_offset, 0, x_offset + img.size[0], img.size[1]))
            x_offset += img.size[0]
            # create uv map
            f_uv = ((self.crops['foreground_uvs'][frame].reshape(-1, 2) * 0.5 + 0.5) * res) #.int()
            input_image = np.array(f_value)
            W, H = input_image.shape[1], input_image.shape[0]
            
            # use alpha instead of mask
            mask = self.crops['foreground_alpha'][frame][0].reshape(1, H, W)
            
            f_value = transforms.ToTensor()(f_value).cuda() # C x H x W
            f_value = f_value * mask
            f_img.append([f_value, mask])
            
            # render img in channel
            for c in range(3):
                interpolated = scipy.interpolate.griddata(
                    points=f_uv.cpu().numpy(),
                    values=f_value[c].reshape(-1, 1).cpu().numpy(),
                    xi=indices.reshape(-1, 2).cpu().numpy(),
                    method="linear",
                ).reshape(res, res)
                interpolated = torch.from_numpy(interpolated).float()
                interpolated[interpolated.isnan()] = 0.0
                f_atlas[frame, :, :, c] = interpolated
                del interpolated
        
        # aggregate
        f_atlas = f_atlas.permute(0, 3, 2, 1) # F x C x W x H
        aggr_atlas, _ = torch.median(f_atlas, dim=0) # mask before atlas
        
        shutil.rmtree('debug')
        os.makedirs('debug')
        
        for i, frame in enumerate(keyframes):
            origin_frame = img_list[i]
            origin_frame.save(f'debug/origin_{frame}.png')
            
            edit_frame = f_img[i][0]
            transforms.ToPILImage()(edit_frame).save(f'debug/edit_{frame}.png')
            
            alpha = f_img[i][1]
            transforms.ToPILImage()(alpha).save(f'debug/mask_{frame}.png')
            
            rec_img = F.grid_sample(aggr_atlas.unsqueeze(0), self.crops['foreground_uvs'][i].reshape(1, -1, 1, 2), mode="bilinear", align_corners=self.data.config["align_corners"]).clamp(min=0.0, max=1.0).reshape(f_img[i][0].shape)
            transforms.ToPILImage()(rec_img).save(f'debug/rec_{frame}.png')
            
            show_atlas, _ = get_atlas_bounding_box(self.data.mask_boundaries, f_atlas[i], self.data.foreground_all_uvs)
            transforms.ToPILImage()(show_atlas).save(f'debug/atlas_{frame}.png')

        aggr_atlas, _ = get_atlas_bounding_box(self.data.mask_boundaries, aggr_atlas, self.data.foreground_all_uvs)
        self.f_atlas = transforms.ToPILImage()(aggr_atlas)
        self.f_atlas.save(f'debug/atlas.png')
        
        return self.f_atlas


if __name__ == '__main__':
    videoe = VideoE(base_cfg="/home/syh/ControlNet/models/cldm_v15.yaml",
                    canny_model_cfg="/home/syh/ControlNet/models/control_sd15_canny.pth",
                    depth_model_cfg="/home/syh/ControlNet/models/control_sd15_depth.pth")
    videoe.load_canny_model()
    videoe.load_depth_model()
    # videoe.load_img2img_model()
    
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
                                      label="Select Example Videos",
                                      value="car-turn")
                load_video_button = gr.Button("Load Video")
                gr.Markdown("### Step 2. write text prompt and advanced options for background and foreground.")
                with gr.Row():
                    f_prompt = gr.Textbox(label="Foreground Prompt", value="a picture of an orange suv")
                    b_prompt = gr.Textbox(label="Background Prompt", value="winter scene, snowy scene, beautiful snow")
                with gr.Row():
                    # with gr.Accordion("Foreground Options", open=False):
                    #     f_image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=256)
                    #     f_low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                    #     f_high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                    #     f_ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                    #     f_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                    #     f_seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                    #     f_eta = gr.Number(label="eta (DDIM)", value=0.0)
                    #     f_a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, no background')
                    #     f_n_prompt = gr.Textbox(label="Negative Prompt",
                    #                         value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
                    with gr.Accordion("Advanced Foreground Options", open=False):
                        adv_keyframes = gr.Textbox(label="keyframe", value="20, 40, 60")
                        adv_atlas_resolution = gr.Slider(label="Atlas Resolution", minimum=1000, maximum=3000, value=2000, step=100)
                        adv_image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=256)
                        adv_low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                        adv_high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                        adv_ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                        adv_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=15.0, value=9.0, step=0.1)
                        adv_seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        adv_eta = gr.Number(label="eta (DDIM)", value=0.0)
                        adv_a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, no background')
                        adv_n_prompt = gr.Textbox(label="Negative Prompt",
                                            value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
                
                    with gr.Accordion("Background Options", open=False):
                        b_image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=256)
                        b_detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=512, step=1)
                        b_ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                        b_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        b_seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        b_eta = gr.Number(label="eta (DDIM)", value=0.0)
                        b_a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                        b_n_prompt = gr.Textbox(label="Negative Prompt",
                                            value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
                gr.Markdown("### Step 3. edit each one and render.")
                with gr.Row():
                    # f_run_button = gr.Button("Edit Foreground")
                    f_advance_run_button = gr.Button("Advanced Edit Foreground (slower, better)")
                    b_run_button = gr.Button("Edit Background")
                run_button = gr.Button("Render")
            with gr.Column():
                output_video = gr.Video(label="Output Video", interactive=False)
                output_foreground_atlas = gr.Image(label="Output Foreground Atlas", type="pil", interactive=False)
                output_background_atlas = gr.Image(label="Output Background Atlas", type="pil", interactive=False)
        
        # edit param
        # f_edit_param = [f_prompt, f_a_prompt, f_n_prompt, f_image_resolution, f_ddim_steps, f_scale, f_seed, f_eta, f_low_threshold, f_high_threshold]
        f_adv_edit_param = [adv_keyframes, adv_atlas_resolution, f_prompt, adv_a_prompt, adv_n_prompt, adv_image_resolution, adv_ddim_steps, adv_scale, adv_seed, adv_eta, adv_low_threshold, adv_high_threshold]
        b_edit_param = [b_prompt, b_a_prompt, b_n_prompt, b_image_resolution, b_detect_resolution, b_ddim_steps, b_scale, b_seed, b_eta]
        # action
        load_video_button.click(fn=videoe.load_video, inputs=video_name, outputs=[original_video, foreground_atlas, background_atlas])
        # f_run_button.click(fn=videoe.edit_foreground, inputs=f_edit_param, outputs=[output_foreground_atlas])
        f_advance_run_button.click(fn=videoe.advanced_edit_foreground, inputs=f_adv_edit_param, outputs=[output_foreground_atlas])
        b_run_button.click(fn=videoe.edit_background, inputs=b_edit_param, outputs=[output_background_atlas])
        run_button.click(fn=videoe.render, outputs=[output_video])
    block.launch(share=True)