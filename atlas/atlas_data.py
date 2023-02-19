import random
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

from atlas.atlas_utils import (
    load_neural_atlases_models,
    get_frames_data,
    get_high_res_atlas,
    get_atlas_crops,
    reconstruct_video_layer,
    create_uv_mask,
    get_masks_boundaries,
    get_random_crop_params,
    get_atlas_bounding_box,
    load_video
)

class AtlasData():
    def __init__(self, video_name):
        with open(f"data/{video_name}/config.json", "r") as f:
            json_dict = json.load(f)
        try:
            maximum_number_of_frames = json_dict["number_of_frames"]
        except:
            maximum_number_of_frames = json_dict["maximum_number_of_frames"]

        config = {
            "device": "cuda",
            "checkpoint_path": f"data/{video_name}/checkpoint",
            "resx": json_dict["resx"],
            "resy": json_dict["resy"],
            "maximum_number_of_frames": maximum_number_of_frames,
            "return_atlas_alpha": False,
            "grid_atlas_resolution": 2000,
            "num_scales": 7,
            "masks_border_expansion": 30,
            "mask_alpha_threshold": 0.99, # 0.95
            "align_corners": False
        }
        self.config = config
        self.device = config["device"]

        self.min_size = min(self.config["resx"], self.config["resy"])
        self.max_size = max(self.config["resx"], self.config["resy"])
        data_folder = f"data/{video_name}/{video_name}"
        self.original_video = load_video(
            data_folder,
            resize=(self.config["resy"], self.config["resx"]),
            num_frames=self.config["maximum_number_of_frames"],
        )
        self.original_video = self.original_video.to(self.device) # tensor

        (
            foreground_mapping,
            background_mapping,
            foreground_atlas_model,
            background_atlas_model,
            alpha_model,
        ) = load_neural_atlases_models(config)
        (
            original_background_all_uvs,
            original_foreground_all_uvs,
            self.all_alpha,
            foreground_atlas_alpha,
        ) = get_frames_data(
            config,
            foreground_mapping,
            background_mapping,
            alpha_model,
        )

        self.background_reconstruction = reconstruct_video_layer(original_background_all_uvs, background_atlas_model)
        # using original video for the foreground layer
        self.foreground_reconstruction = self.original_video * self.all_alpha

        (
            self.background_all_uvs,
            self.scaled_background_uvs,
            self.background_min_u,
            self.background_min_v,
            self.background_max_u,
            self.background_max_v,
        ) = self.preprocess_uv_values(
            original_background_all_uvs, config["grid_atlas_resolution"], device=self.device, layer="background"
        )
        (
            self.foreground_all_uvs,
            self.scaled_foreground_uvs,
            self.foreground_min_u,
            self.foreground_min_v,
            self.foreground_max_u,
            self.foreground_max_v,
        ) = self.preprocess_uv_values(
            original_foreground_all_uvs, config["grid_atlas_resolution"], device=self.device, layer="foreground"
        )

        self.background_uv_mask = create_uv_mask(
            config,
            background_mapping,
            self.background_min_u,
            self.background_min_v,
            self.background_max_u,
            self.background_max_v,
            uv_shift=-0.5,
            resolution_shift=1,
        )
        self.foreground_uv_mask = create_uv_mask(
            config,
            foreground_mapping,
            self.foreground_min_u,
            self.foreground_min_v,
            self.foreground_max_u,
            self.foreground_max_v,
            uv_shift=0.5,
            resolution_shift=0,
        )
        self.background_grid_atlas = get_high_res_atlas(
            background_atlas_model,
            self.background_min_v,
            self.background_min_u,
            self.background_max_v,
            self.background_max_u,
            config["grid_atlas_resolution"],
            device=config["device"],
            layer="background",
        )
        self.foreground_grid_atlas = get_high_res_atlas(
            foreground_atlas_model,
            self.foreground_min_v,
            self.foreground_min_u,
            self.foreground_max_v,
            self.foreground_max_u,
            config["grid_atlas_resolution"],
            device=config["device"],
            layer="foreground",
        )
        if config["return_atlas_alpha"]:
            self.foreground_atlas_alpha = foreground_atlas_alpha  # used for visualizations
        self.cnn_min_crop_size = 2 ** self.config["num_scales"] + 1

        self.mask_boundaries = get_masks_boundaries(
            alpha_video=self.all_alpha.cpu(),
            border=self.config["masks_border_expansion"],
            threshold=self.config["mask_alpha_threshold"],
            min_crop_size=self.cnn_min_crop_size,
        )
        self.cropped_foreground_atlas, self.foreground_atlas_bbox = get_atlas_bounding_box(
            self.mask_boundaries, self.foreground_grid_atlas, self.foreground_all_uvs
        )

        self.step = -1

        crop_transforms = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)],
                    p=0.1,
                ),
            ]
        )
        self.crop_aug = crop_transforms
        self.edited_atlas_dict, self.edit_dict, self.uv_mask = {}, {}, {}
    @staticmethod
    def preprocess_uv_values(layer_uv_values, resolution, device="cuda", layer="background"):
        if layer == "background":
            shift = 1
        else:
            shift = 0
        uv_values = (layer_uv_values + shift) * resolution
        min_u, min_v = uv_values.reshape(-1, 2).min(dim=0).values.long()
        uv_values -= torch.tensor([min_u, min_v], device=device)
        max_u, max_v = uv_values.reshape(-1, 2).max(dim=0).values.ceil().long()

        edge_size = torch.tensor([max_u, max_v], device=device)
        scaled_uv_values = ((uv_values.reshape(-1, 2) / edge_size) * 2 - 1).unsqueeze(1).unsqueeze(0)

        return uv_values, scaled_uv_values, min_u, min_v, max_u, max_v

    def get_random_crop_data(self, crop_size):
        t = random.randint(0, self.config["maximum_number_of_frames"] - 1)
        y_start, x_start, h_crop, w_crop = get_random_crop_params((self.config["resx"], self.config["resy"]), crop_size)
        return y_start, x_start, h_crop, w_crop, t

    def get_global_crops_multi(self):
        foreground_atlas_crops = []
        background_atlas_crops = []
        foreground_uvs = []
        background_uvs = []
        background_alpha_crops = []
        foreground_alpha_crops = []
        original_background_crops = []
        original_foreground_crops = []
        output_dict = {}

        self.config["crops_min_cover"] = 0.95
        self.config["grid_atlas_resolution"] = 2000
        self.config["center_frame_distance"] = 16
        self.dist = self.config["center_frame_distance"]
        
        t = random.randint(self.dist, self.config["maximum_number_of_frames"] - 1 - self.dist)
        flip = False

        # TODO: keyframe list should be indicted by user
        # keyframes_list = [0, self.config["maximum_number_of_frames"] // 2, self.config["maximum_number_of_frames"] - 1]
        # for cur_frame in keyframes_list:   
        
        for cur_frame in [t - self.dist, t, t + self.dist]:
            y_start, x_start, frame_h, frame_w = self.mask_boundaries[cur_frame].tolist()
            crop_size = (
                max(
                    random.randint(round(self.config["crops_min_cover"] * frame_h), frame_h),
                    self.cnn_min_crop_size,
                ),
                max(
                    random.randint(round(self.config["crops_min_cover"] * frame_w), frame_w),
                    self.cnn_min_crop_size,
                ),
            )
            y_crop, x_crop, h_crop, w_crop = get_random_crop_params((frame_w, frame_h), crop_size)

            foreground_uv = self.foreground_all_uvs[
                cur_frame,
                y_start + y_crop : y_start + y_crop + h_crop,
                x_start + x_crop : x_start + x_crop + w_crop,
            ]
            alpha = self.all_alpha[
                [cur_frame],
                :,
                y_start + y_crop : y_start + y_crop + h_crop,
                x_start + x_crop : x_start + x_crop + w_crop,
            ]

            original_foreground_crop = self.foreground_reconstruction[
                [cur_frame],
                :,
                y_start + y_crop : y_start + y_crop + h_crop,
                x_start + x_crop : x_start + x_crop + w_crop,
            ]

            original_foreground_crop = self.crop_aug(original_foreground_crop)
            foreground_alpha_crops.append(alpha.flip(-1) if flip else alpha)
            foreground_uvs.append(foreground_uv)  # not scaled
            original_foreground_crops.append(
                original_foreground_crop.flip(-1) if flip else original_foreground_crop
            )
        import pdb; pdb.set_trace()
        foreground_max_vals = torch.tensor(
            [self.config["grid_atlas_resolution"]] * 2, device=self.device, dtype=torch.long
        )
        foreground_min_vals = torch.tensor([0] * 2, device=self.device, dtype=torch.long)
        for uv_values in foreground_uvs:
            min_uv = uv_values.amin(dim=[0, 1]).long()
            max_uv = uv_values.amax(dim=[0, 1]).ceil().long()
            foreground_min_vals = torch.minimum(foreground_min_vals, min_uv)
            foreground_max_vals = torch.maximum(foreground_max_vals, max_uv)
        
        h_v = foreground_max_vals[1] - foreground_min_vals[1]
        w_u = foreground_max_vals[0] - foreground_min_vals[0]
        foreground_atlas_crop = crop(
            self.foreground_grid_atlas,
            foreground_min_vals[1],
            foreground_min_vals[0],
            h_v,
            w_u,
        )
        foreground_atlas_crop = self.crop_aug(foreground_atlas_crop)

        for i, uv_values in enumerate(foreground_uvs):
            foreground_uvs[i] = (
                2 * (uv_values - foreground_min_vals) / (foreground_max_vals - foreground_min_vals) - 1
            ).unsqueeze(0)
        foreground_atlas_crops.append(foreground_atlas_crop.flip(-1) if flip else foreground_atlas_crop)


        crop_size = (
            random.randint(round(self.config["crops_min_cover"] * self.min_size), self.min_size),
            random.randint(round(self.config["crops_min_cover"] * self.max_size), self.max_size),
        )
        crop_data = self.get_random_crop_data(crop_size)
        y, x, h, w, _ = crop_data
        background_uv = self.background_all_uvs[[t - self.dist, t, t + self.dist], y : y + h, x : x + w]
        original_background_crop = self.background_reconstruction[
            [t - self.dist, t, t + self.dist], :, y : y + h, x : x + w
        ]
        alpha = self.all_alpha[[t - self.dist, t, t + self.dist], :, y : y + h, x : x + w]

        original_background_crop = self.crop_aug(original_background_crop)

        original_background_crops = [
            el.unsqueeze(0).flip(-1) if flip else el.unsqueeze(0) for el in original_background_crop
        ]
        background_alpha_crops = [el.unsqueeze(0).flip(-1) if flip else el.unsqueeze(0) for el in alpha]

        background_atlas_crop, background_min_vals, background_max_vals = get_atlas_crops(
            background_uv,
            self.background_grid_atlas,
            self.crop_aug,
        )
        background_uv = 2 * (background_uv - background_min_vals) / (background_max_vals - background_min_vals) - 1
        if flip:
            background_uv[:, :, :, 0] = -background_uv[:, :, :, 0]
            background_uv = background_uv.flip(-2)
        background_atlas_crops = [
            el.unsqueeze(0).flip(-1) if flip else el.unsqueeze(0) for el in background_atlas_crop
        ]
        background_uvs = [el.unsqueeze(0) for el in background_uv]

        output_dict["foreground_alpha"] = foreground_alpha_crops
        output_dict["foreground_uvs"] = foreground_uvs
        output_dict["original_foreground_crops"] = original_foreground_crops
        output_dict["foreground_atlas_crops"] = foreground_atlas_crops

        output_dict["background_alpha"] = background_alpha_crops
        output_dict["background_uvs"] = background_uvs
        output_dict["original_background_crops"] = original_background_crops
        output_dict["background_atlas_crops"] = background_atlas_crops
        return output_dict