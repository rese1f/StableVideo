from PIL import Image
from pathlib import Path
import scipy.interpolate
import torch
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm import tqdm
import numpy as np
import cv2

from stablevideo.implicit_neural_networks import IMLP


def load_video(folder: str, resize=(432, 768), num_frames=70):
    resy, resx = resize
    folder = Path(folder)
    input_files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))[:num_frames]
    video_tensor = torch.zeros((len(input_files), 3, resy, resx))
    
    for i, file in enumerate(input_files):
        video_tensor[i] = transforms.ToTensor()(Image.open(str(file)).resize((resx, resy), Image.LANCZOS))
        
    return video_tensor


def load_neural_atlases_models(config):
    foreground_mapping = IMLP(
        input_dim=3,
        output_dim=2,
        hidden_dim=256,
        use_positional=False,
        num_layers=6,
        skip_layers=[],
    ).to(config["device"])

    background_mapping = IMLP(
        input_dim=3,
        output_dim=2,
        hidden_dim=256,
        use_positional=False,
        num_layers=4,
        skip_layers=[],
    ).to(config["device"])

    foreground_atlas_model = IMLP(
        input_dim=2,
        output_dim=3,
        hidden_dim=256,
        use_positional=True,
        positional_dim=10,
        num_layers=8,
        skip_layers=[4, 7],
    ).to(config["device"])

    background_atlas_model = IMLP(
        input_dim=2,
        output_dim=3,
        hidden_dim=256,
        use_positional=True,
        positional_dim=10,
        num_layers=8,
        skip_layers=[4, 7],
    ).to(config["device"])

    alpha_model = IMLP(
        input_dim=3,
        output_dim=1,
        hidden_dim=256,
        use_positional=True,
        positional_dim=5,
        num_layers=8,
        skip_layers=[],
    ).to(config["device"])

    checkpoint = torch.load(config["checkpoint_path"])
    foreground_mapping.load_state_dict(checkpoint["model_F_mapping1_state_dict"])
    background_mapping.load_state_dict(checkpoint["model_F_mapping2_state_dict"])
    foreground_atlas_model.load_state_dict(checkpoint["F_atlas_state_dict"])
    background_atlas_model.load_state_dict(checkpoint["F_atlas_state_dict"])
    alpha_model.load_state_dict(checkpoint["model_F_alpha_state_dict"])

    foreground_mapping = foreground_mapping.eval().requires_grad_(False)
    background_mapping = background_mapping.eval().requires_grad_(False)
    foreground_atlas_model = foreground_atlas_model.eval().requires_grad_(False)
    background_atlas_model = background_atlas_model.eval().requires_grad_(False)
    alpha_model = alpha_model.eval().requires_grad_(False)

    return foreground_mapping, background_mapping, foreground_atlas_model, background_atlas_model, alpha_model


@torch.no_grad()
def get_frames_data(config, foreground_mapping, background_mapping, alpha_model):
    max_size = max(config["resx"], config["resy"])
    normalizing_factor = torch.tensor([max_size / 2, max_size / 2, config["maximum_number_of_frames"] / 2])
    background_uv_values = torch.zeros(
        size=(config["maximum_number_of_frames"], config["resy"], config["resx"], 2), device=config["device"]
    )
    foreground_uv_values = torch.zeros(
        size=(config["maximum_number_of_frames"], config["resy"], config["resx"], 2), device=config["device"]
    )
    alpha = torch.zeros(
        size=(config["maximum_number_of_frames"], config["resy"], config["resx"], 1), device=config["device"]
    )

    for frame in tqdm(range(config["maximum_number_of_frames"]), leave=False):
        indices = get_grid_indices(0, 0, config["resy"], config["resx"], t=torch.tensor(frame))

        normalized_chunk = (indices / normalizing_factor - 1).to(config["device"])

        # get the atlas UV coordinates from the two mapping networks;
        with torch.no_grad():
            current_background_uv_values = background_mapping(normalized_chunk)
            current_foreground_uv_values = foreground_mapping(normalized_chunk)
            current_alpha = alpha_model(normalized_chunk)

        background_uv_values[frame, indices[:, 1], indices[:, 0]] = current_background_uv_values * 0.5 - 0.5
        foreground_uv_values[frame, indices[:, 1], indices[:, 0]] = current_foreground_uv_values * 0.5 + 0.5
        current_alpha = 0.5 * (current_alpha + 1.0)
        current_alpha = 0.99 * current_alpha + 0.001
        alpha[frame, indices[:, 1], indices[:, 0]] = current_alpha
    # config["return_atlas_alpha"] = True
    if config["return_atlas_alpha"]:  # this should take a few minutes
        foreground_atlas_alpha = torch.zeros(
            size=(
                config["maximum_number_of_frames"],
                config["grid_atlas_resolution"],
                config["grid_atlas_resolution"],
                1,
            ),
        )
        # foreground_uv_values: 70 x 432 x 768 x 2
        foreground_uv_values_grid = foreground_uv_values * config["grid_atlas_resolution"]
        # indices: 4000000 x 2
        indices = get_grid_indices(0, 0, config["grid_atlas_resolution"], config["grid_atlas_resolution"])
        for frame in tqdm(range(config["maximum_number_of_frames"]), leave=False):
            interpolated = scipy.interpolate.griddata(
                foreground_uv_values_grid[frame].reshape(-1, 2).cpu().numpy(), # 432 x 768 x 2 -> -1 x 2
                alpha[frame]
                .reshape(
                    -1,
                )
                .cpu()
                .numpy(),
                indices.reshape(-1, 2).cpu().numpy(),
                method="linear",
            ).reshape(config["grid_atlas_resolution"], config["grid_atlas_resolution"], 1)
            foreground_atlas_alpha[frame] = torch.from_numpy(interpolated)
        foreground_atlas_alpha[foreground_atlas_alpha.isnan()] = 0.0
        foreground_atlas_alpha = (
            torch.median(foreground_atlas_alpha, dim=0, keepdim=True).values.to(config["device"]).permute(0, 3, 2, 1)
        )
    else:
        foreground_atlas_alpha = None
    return background_uv_values, foreground_uv_values, alpha.permute(0, 3, 1, 2), foreground_atlas_alpha


@torch.no_grad()
def reconstruct_video_layer(uv_values, atlas_model):
    t, h, w, _ = uv_values.shape
    reconstruction = torch.zeros(size=(t, h, w, 3), device=uv_values.device)
    for frame in range(t):
        rgb = (atlas_model(uv_values[frame].reshape(-1, 2)) + 1) * 0.5
        reconstruction[frame] = rgb.reshape(h, w, 3)
    return reconstruction.permute(0, 3, 1, 2)


@torch.no_grad()
def create_uv_mask(config, mapping_model, min_u, min_v, max_u, max_v, uv_shift=-0.5, resolution_shift=1):
    max_size = max(config["resx"], config["resy"])
    normalizing_factor = torch.tensor([max_size / 2, max_size / 2, config["maximum_number_of_frames"] / 2])
    resolution = config["grid_atlas_resolution"]
    uv_mask = torch.zeros(size=(resolution, resolution), device=config["device"])

    for frame in tqdm(range(config["maximum_number_of_frames"]), leave=False):
        indices = get_grid_indices(0, 0, config["resy"], config["resx"], t=torch.tensor(frame))
        for chunk in indices.split(50000, dim=0):
            normalized_chunk = (chunk / normalizing_factor - 1).to(config["device"])

            # get the atlas UV coordinates from the two mapping networks;
            with torch.no_grad():
                uv_values = mapping_model(normalized_chunk)
            uv_values = uv_values * 0.5 + uv_shift
            uv_values = ((uv_values + resolution_shift) * resolution).clip(0, resolution - 1)

            uv_mask[uv_values[:, 1].floor().long(), uv_values[:, 0].floor().long()] = 1
            uv_mask[uv_values[:, 1].floor().long(), uv_values[:, 0].ceil().long()] = 1
            uv_mask[uv_values[:, 1].ceil().long(), uv_values[:, 0].floor().long()] = 1
            uv_mask[uv_values[:, 1].ceil().long(), uv_values[:, 0].ceil().long()] = 1

    uv_mask = crop(uv_mask.unsqueeze(0).unsqueeze(0), min_v, min_u, max_v, max_u)
    return uv_mask.detach().cpu()  # shape [1, 1, resolution, resolution]


@torch.no_grad()
def get_high_res_atlas(atlas_model, min_v, min_u, max_v, max_u, resolution, device="cuda", layer="background"):
    inds_grid = get_grid_indices(0, 0, resolution, resolution)
    inds_grid_chunks = inds_grid.split(50000, dim=0)
    if layer == "background":
        shift = -1
    else:
        shift = 0

    rendered_atlas = torch.zeros((resolution, resolution, 3)).to(device)  # resy, resx, 3
    with torch.no_grad():
        # reconstruct image row by row
        for chunk in inds_grid_chunks:
            normalized_chunk = torch.stack(
                [
                    (chunk[:, 0] / resolution) + shift,
                    (chunk[:, 1] / resolution) + shift,
                ],
                dim=-1,
            ).to(device)

            rgb_output = atlas_model(normalized_chunk)
            rendered_atlas[chunk[:, 1], chunk[:, 0], :] = rgb_output
        # move colors to RGB color domain (0,1)
    rendered_atlas = 0.5 * (rendered_atlas + 1)
    rendered_atlas = rendered_atlas.permute(2, 0, 1).unsqueeze(0)  # shape (1, 3, resy, resx)
    cropped_atlas = crop(
        rendered_atlas,
        min_v,
        min_u,
        max_v,
        max_u,
    )

    return cropped_atlas


def get_grid_indices(x_start, y_start, h_crop, w_crop, t=None):
    crop_indices = torch.meshgrid(torch.arange(w_crop) + x_start, torch.arange(h_crop) + y_start)
    crop_indices = torch.stack(crop_indices, dim=-1)
    crop_indices = crop_indices.reshape(h_crop * w_crop, crop_indices.shape[-1])
    if t is not None:
        crop_indices = torch.cat([crop_indices, t.repeat(h_crop * w_crop, 1)], dim=1)
    return crop_indices


def get_atlas_crops(uv_values, grid_atlas, augmentation=None):
    if len(uv_values.shape) == 3:
        dims = [0, 1]
    elif len(uv_values.shape) == 4:
        dims = [0, 1, 2]
    else:
        raise ValueError("uv_values should be of shape of len 3 or 4")

    min_u, min_v = uv_values.amin(dim=dims).long()
    max_u, max_v = uv_values.amax(dim=dims).ceil().long()
    # min_u, min_v = uv_values.min(dim=0).values
    # max_u, max_v = uv_values.max(dim=0).values

    h_v = max_v - min_v
    w_u = max_u - min_u
    atlas_crop = crop(grid_atlas, min_v, min_u, h_v, w_u)
    if augmentation is not None:
        atlas_crop = augmentation(atlas_crop)
    return atlas_crop, torch.stack([min_u, min_v]), torch.stack([max_u, max_v])


def get_random_crop_params(input_size, output_size):
    w, h = input_size
    th, tw = output_size

    if h + 1 < th or w + 1 < tw:
        raise ValueError(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")

    if w == tw and h == th:
        return 0, 0, h, w

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()
    return i, j, th, tw


def get_masks_boundaries(alpha_video, border=20, threshold=0.95, min_crop_size=2 ** 7 + 1):
    resy, resx = alpha_video.shape[-2:]
    num_frames = alpha_video.shape[0]
    masks_borders = torch.zeros((num_frames, 4), dtype=torch.int64)
    for i, file in enumerate(range(num_frames)):
        mask_im = alpha_video[i]
        mask_im[mask_im >= threshold] = 1
        mask_im[mask_im < threshold] = 0
        all_ones = mask_im.squeeze().nonzero()
        min_y, min_x = torch.maximum(all_ones.min(dim=0).values - border, torch.tensor([0, 0]))
        max_y, max_x = torch.minimum(all_ones.max(dim=0).values + border, torch.tensor([resy, resx]))
        h = max_y - min_y
        w = max_x - min_x
        if h < min_crop_size:
            pad = min_crop_size - h
            if max_y + pad > resy:
                min_y -= pad
            else:
                max_y += pad
            h = max_y - min_y
        if w < min_crop_size:
            pad = min_crop_size - w
            if max_x + pad > resx:
                min_x -= pad
            else:
                max_x += pad
            w = max_x - min_x
        masks_borders[i] = torch.tensor([min_y, min_x, h, w])
    return masks_borders


def get_atlas_bounding_box(mask_boundaries, grid_atlas, video_uvs):
    min_uv = torch.tensor(grid_atlas.shape[-2:], device=video_uvs.device)
    max_uv = torch.tensor([0, 0], device=video_uvs.device)
    for boundary, frame in zip(mask_boundaries, video_uvs):
        cropped_uvs = crop(frame.permute(2, 0, 1).unsqueeze(0), *list(boundary))  # 1,2,h,w
        min_uv = torch.minimum(cropped_uvs.amin(dim=[0, 2, 3]), min_uv).floor().int()
        max_uv = torch.maximum(cropped_uvs.amax(dim=[0, 2, 3]), max_uv).ceil().int()

    hw = max_uv - min_uv
    crop_data = [*list(min_uv)[::-1], *list(hw)[::-1]]
    return crop(grid_atlas, *crop_data), crop_data


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)