"""Build aligned Huawei camera-display training pairs from capture triplets."""

import os
import cv2
import rawpy
import torch

import json
import argparse
import subprocess
import numpy as np

from tqdm import tqdm
from pathlib import Path
from os import path as osp
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from fractions import Fraction
from raft.core.flow import FLOW
from utils.metadata import get_metadata
from utils.isp_pipeline import run_pipeline
from utils.metrics import calculate_psnr, calculate_ssim
from utils.img import warp_image, imwrite, tensor2img
from utils.denoising import Raw_Dn


def _parse_exposure_time(exposure_str):
    """Convert Exif exposure text such as ``1/50`` to a float."""
    if not exposure_str:
        return 0.0
    try:
        return float(Fraction(exposure_str))
    except (ValueError, ZeroDivisionError):
        return 0.0

def fix_orientation(image, orientation):
    """Apply the Exif orientation transform represented by values 1 through 8."""

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image

def crop_image_patch(image_to_crop, display_shape, center_x, center_y,
                     target_height, multiple_of=16, draw_box=False):
    """Crop a display-aspect-ratio ROI and optionally return a boxed preview."""
    display_h, display_w, _ = display_shape

    patch_height = int(target_height)
    patch_width = int(display_w * patch_height / display_h)
    patch_width = (patch_width // multiple_of) * multiple_of
    patch_height = (patch_height // multiple_of) * multiple_of

    half_w = patch_width // 2
    half_h = patch_height // 2

    x1 = center_x - half_w
    y1 = center_y - half_h
    x2 = x1 + patch_width
    y2 = y1 + patch_height

    if draw_box:
        image_with_box = image_to_crop.copy()
        cv2.rectangle(image_with_box, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
    else:
        image_with_box = None
    
    image_to_crop = image_to_crop[y1:y2, x1:x2, :]

    return image_to_crop, image_with_box

def _roi_bbox_from_center(display_h, display_w, center_x, center_y, target_height, multiple_of=16):
    """Return the ROI box used by ``crop_image_patch``."""
    patch_h = int(target_height)
    patch_w = int(display_w * patch_h / display_h)

    patch_w = (patch_w // multiple_of) * multiple_of
    patch_h = (patch_h // multiple_of) * multiple_of

    half_w = patch_w // 2
    half_h = patch_h // 2

    x1 = int(center_x - half_w)
    y1 = int(center_y - half_h)
    x2 = x1 + patch_w
    y2 = y1 + patch_h
    return x1, y1, x2, y2


def _expand_bbox_to_multiple(x1, y1, x2, y2, H, W, m=256):
    """Expand a box to multiples of ``m`` while keeping it in the image."""
    ex1 = (x1 // m) * m
    ey1 = (y1 // m) * m
    ex2 = ((x2 + m - 1) // m) * m
    ey2 = ((y2 + m - 1) // m) * m

    ew = ex2 - ex1
    eh = ey2 - ey1

    # Fall back to the full extent if a multiple-sized box cannot fit.
    if ew > W:
        ex1, ex2 = 0, W
    else:
        if ex2 > W:
            shift = ex2 - W
            ex1 -= shift
            ex2 -= shift
        if ex1 < 0:
            shift = -ex1
            ex1 += shift
            ex2 += shift

    if eh > H:
        ey1, ey2 = 0, H
    else:
        if ey2 > H:
            shift = ey2 - H
            ey1 -= shift
            ey2 -= shift
        if ey1 < 0:
            shift = -ey1
            ey1 += shift
            ey2 += shift

    ex1 = int(max(0, min(ex1, W)))
    ex2 = int(max(0, min(ex2, W)))
    ey1 = int(max(0, min(ey1, H)))
    ey2 = int(max(0, min(ey2, H)))
    return ex1, ey1, ex2, ey2


def _mean_blur_rgb(array, kernel_size):
    """Apply the paper's depthwise display blur while preserving dtype."""
    if array.ndim != 3 or kernel_size % 2 != 1:
        raise ValueError("Expected an HWC image and an odd blur kernel")
    source_dtype = array.dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = (
        torch.from_numpy(np.ascontiguousarray(array))
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    channels = tensor.shape[1]
    kernel = torch.ones(
        (channels, 1, kernel_size, kernel_size), device=device
    ) / (kernel_size * kernel_size)
    result = F.conv2d(
        tensor, kernel, padding=kernel_size // 2, groups=channels
    )[0].permute(1, 2, 0).cpu().numpy()
    if np.issubdtype(source_dtype, np.integer):
        limits = np.iinfo(source_dtype)
        return np.rint(result).clip(limits.min, limits.max).astype(source_dtype)
    return result.astype(source_dtype, copy=False)


torch.backends.cudnn.benchmark = True

def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help="Folder containing Camera_Raw, Camera_ISP and Display_sRGB")
    parser.add_argument('--gainmap', required=True, help="Path to gainmaps NPZ")
    parser.add_argument('--gain_type', default="gain_center1", help="gain_center1, gain_edge1")
    parser.add_argument('--output', default="Final_Pair_0.5x", help="Legacy output name below --dataset")
    parser.add_argument('--output-root', default=None, help="Explicit output directory; preferred for packaged datasets")
    parser.add_argument('--base_exposure', type=float, default=0.25, help="Reference exposure used to normalize captures")
    parser.add_argument('--flow-checkpoint', required=True, help="RAFT/FLOW checkpoint")
    parser.add_argument('--denoise-checkpoint', required=True, help="DualDn RAW checkpoint")
    parser.add_argument('--flat-raw', required=True, help="Subtractive RAW dark-frame NPY")
    parser.add_argument('--flat-rgb', required=True, help="Subtractive ISP dark-frame PNG")
    parser.add_argument('--raw-roi', type=float, nargs=3, default=(2056, 1444, 1026), metavar=('X', 'Y', 'HEIGHT'))
    parser.add_argument('--isp-roi', type=float, nargs=3, default=(2055, 1441, 1131), metavar=('X', 'Y', 'HEIGHT'))
    parser.add_argument('--resize-scale', type=float, default=0.5)
    parser.add_argument('--border-crop', type=int, default=16)
    parser.add_argument('--flow-iterations', type=int, default=40)
    parser.add_argument('--blur-display-kernel', type=int, default=3)
    parser.add_argument('--process-images', action=argparse.BooleanOptionalAction, default=True)

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    center_x_Raw, center_y_Raw, absolute_h_Raw = args.raw_roi
    center_x_ISP, center_y_ISP, absolute_h_ISP = args.isp_roi
    scale = args.resize_scale
    blur_size = args.blur_display_kernel
    amplification_ratio = 1
    cropping_border = args.border_crop
    save_process_img = args.process_images

    DatasetFolder = args.dataset
    DisplaysRGB_Folder = osp.join(DatasetFolder, 'Display_sRGB')
    CameraRaw_Folder = osp.join(DatasetFolder, 'Camera_Raw')
    CameraISP_Folder = osp.join(DatasetFolder, 'Camera_ISP')
    
    files = sorted(os.listdir(DisplaysRGB_Folder))
    
    OutputFolder = args.output_root or osp.join(args.dataset, args.output)
    OutputFolder_Process = osp.join(OutputFolder, "Process")
    OutputFolder_Flow = osp.join(OutputFolder, "Image_Pairs_Flow")
    
    os.makedirs(OutputFolder, exist_ok=True)
    os.makedirs(OutputFolder_Process, exist_ok=True)
    os.makedirs(OutputFolder_Flow, exist_ok=True)

    out_flie_Flow = open(OutputFolder + '/Flow_validation.txt', 'w')
    out_flie_exposure = open(OutputFolder + '/Exposure.txt', 'w')

    print("Loading FLOW model...")
    model = torch.nn.DataParallel(FLOW())
    model.load_state_dict(torch.load(args.flow_checkpoint, map_location='cpu'))
    model = model.module
    model.cuda()
    model.eval()
    print("Model loaded.")

    print("Loading Denoising model...")
    ckpt = torch.load(args.denoise_checkpoint, map_location='cpu')
    sd   = ckpt.get('params', ckpt)
    dn_model = Raw_Dn()
    dn_model.load_state_dict(sd, strict=True)
    dn_model = dn_model.cuda().eval()
    print("Model loaded.")

    for Filename in tqdm(files):
        idx = int(Path(Filename).stem.split('_', 1)[0][:4].zfill(4)) 
        
        DisplaysRGB_Path = osp.join(DisplaysRGB_Folder, Filename)
        CameraRaw_Path = osp.join(CameraRaw_Folder, Filename.replace('DisplaysRGB.png', 'CameraRaw.dng'))
        CameraISP_Path = osp.join(CameraISP_Folder, Filename.replace('DisplaysRGB.png', 'CameraISP.jpg'))
        
        DisplaysRGB = Image.open(DisplaysRGB_Path).convert('RGB')
        to_tensor_converter = transforms.ToTensor()
        DisplaysRGB = to_tensor_converter(DisplaysRGB)
        
        CameraISP = Image.open(CameraISP_Path).convert('RGB')
        to_tensor_converter = transforms.ToTensor()
        CameraISP = to_tensor_converter(CameraISP)
        
        CameraRaw = rawpy.imread(CameraRaw_Path)
        Flip = CameraRaw.sizes.flip
        
        npz = np.load(args.gainmap)[args.gain_type]
        gaimmap = torch.from_numpy(npz).to(torch.float32).clamp_min(1e-8).permute(2,0,1).cuda()
        
        
        exifdata = json.loads(subprocess.run(['exiftool', '-j', CameraRaw_Path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)[0]
        noise_profile = torch.tensor([float(x) for x in exifdata.get('NoiseProfile', None).split()])
        
        Exposure_Time  = torch.tensor(_parse_exposure_time(exifdata.get('ExposureTime')))
        F_Number = torch.tensor(float(exifdata.get('FNumber')))
        ISO = torch.tensor(float(exifdata.get('ISO')))
        EV = torch.tensor(float(exifdata.get('BaselineExposure')))
        current_exposure = (Exposure_Time * ISO * (2 ** EV)) / (F_Number ** 2) 
        relative_exposure = args.base_exposure / current_exposure
        
        print(f"idx:{idx}, current_exposure:{current_exposure}")
        
        out_flie_exposure.write(f"{idx:04d}\t{relative_exposure}\n")
        out_flie_exposure.flush()
        
        
        Camerametadata = get_metadata(CameraRaw)
        CameraRaw = CameraRaw.raw_image_visible.astype(np.float32)

        Camerametadata['alpha'] = 0
        Camerametadata['ref'] = 'D65'
        Camerametadata['demosaic_type'] = 'AHD'
        Camerametadata['color_desc'] =  'RGBG'
        Camerametadata['gamma_type'] = 'Rec709'

        Camerametadata['wb_matrix'] = torch.Tensor(Camerametadata['wb_matrix']).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()
        Camerametadata['color_mask'] = torch.from_numpy(Camerametadata['color_mask']).unsqueeze(0).unsqueeze(0).cuda()
        Camerametadata['rgb_xyz_matrix'] = torch.Tensor(Camerametadata['rgb_xyz_matrix']).unsqueeze(0).cuda()
        
        
        Norm_CameraRaw = run_pipeline(torch.from_numpy(CameraRaw).unsqueeze(0).unsqueeze(0).cuda(), \
                    Camerametadata, 'raw', 'normal')

        display_h = int(DisplaysRGB.shape[1])
        display_w = int(DisplaysRGB.shape[2])

        roi_x1, roi_y1, roi_x2, roi_y2 = _roi_bbox_from_center(
            display_h, display_w,
            center_x_Raw, center_y_Raw,
            absolute_h_Raw,
            multiple_of=16
        )

        H_raw = int(Norm_CameraRaw.shape[-2])
        W_raw = int(Norm_CameraRaw.shape[-1])


        # DualDn requires a patch whose spatial dimensions are multiples of 256.
        dn_x1, dn_y1, dn_x2, dn_y2 = _expand_bbox_to_multiple(
            roi_x1, roi_y1, roi_x2, roi_y2,
            H_raw, W_raw,
            m=256
        )

        # Denoise only the capture ROI and paste it back into the normalized RAW.
        x_patch  = Norm_CameraRaw[:, :, dn_y1:dn_y2, dn_x1:dn_x2]
        cm_patch = Camerametadata['color_mask'][:, :, dn_y1:dn_y2, dn_x1:dn_x2]
        
        with torch.no_grad():
            y_patch  = dn_model(x_patch, colormask=cm_patch, k=float(noise_profile[0]), sigma=float(noise_profile[1]))
        
        Norm_CameraRaw[:, :, dn_y1:dn_y2, dn_x1:dn_x2] = y_patch

        flat_npz = np.load(args.flat_raw).astype(np.float32)
        flat_Raw = torch.from_numpy(flat_npz).to(torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        
        flat_png = Image.open(args.flat_rgb).convert('RGB')
        to_tensor_converter = transforms.ToTensor()
        flat_png = to_tensor_converter(flat_png)
        CameraISP = CameraISP - flat_png
        
        Norm_CameraRaw = (Norm_CameraRaw - flat_Raw).clamp_min(1e-8)

        mid_sRGB = run_pipeline(Norm_CameraRaw*amplification_ratio, \
                    Camerametadata, 'normal', 'demosaic')
        
        mid_sRGB = mid_sRGB * relative_exposure * gaimmap
        
        CamerasRGB = run_pipeline(mid_sRGB, \
                    Camerametadata, 'demosaic', 'gamma').squeeze(0).squeeze(0)
        
        Camerametadata['wb_matrix'] = torch.ones((1, 1, 1, 4), dtype=torch.float32).cuda()
        
        CameraRaw = run_pipeline(Norm_CameraRaw, \
                    Camerametadata, 'normal', 'demosaic')
        
        CameraRaw = CameraRaw * relative_exposure * gaimmap
        
        
        DisplaysRGB = tensor2img(DisplaysRGB, True)
        CameraRaw = tensor2img(CameraRaw, rgb2bgr=False, out_type=np.float64)
        CamerasRGB = tensor2img(CamerasRGB, True)
        CameraISP = tensor2img(CameraISP, True)
        
        CamerasRGB = fix_orientation(CamerasRGB, Flip)
        CameraRaw = fix_orientation(CameraRaw, Flip)


        CameraRaw_ROI, CameraRaw_box = crop_image_patch(
            image_to_crop=CameraRaw,
            display_shape=DisplaysRGB.shape,
            center_x=center_x_Raw,
            center_y=center_y_Raw,
            target_height=absolute_h_Raw,
            draw_box=save_process_img
        )

        CamerasRGB_ROI, CamerasRGB_box = crop_image_patch(
            image_to_crop=CamerasRGB,
            display_shape=DisplaysRGB.shape,
            center_x=center_x_Raw,
            center_y=center_y_Raw,
            target_height=absolute_h_Raw,
            draw_box=save_process_img
        )

        CameraISP_ROI, CameraISP_box = crop_image_patch(
            image_to_crop=CameraISP,
            display_shape=DisplaysRGB.shape,
            center_x=center_x_ISP,
            center_y=center_y_ISP,
            target_height=absolute_h_ISP,
            draw_box=save_process_img
        )
        
        if save_process_img:
            imwrite((cv2.cvtColor(CameraRaw_ROI.astype(np.float32), cv2.COLOR_RGB2BGR) * 255.0).round(), osp.join(OutputFolder_Process, f'{idx:04d}_CameraRaw_Crop.png'))
            imwrite((cv2.cvtColor(CameraRaw_box.astype(np.float32), cv2.COLOR_RGB2BGR) * 255.0).round(), osp.join(OutputFolder_Process, f'{idx:04d}_CameraRaw_CropBox.png'))
            imwrite(CamerasRGB_ROI, osp.join(OutputFolder_Process, f'{idx:04d}_CamerasRGB_Crop.png'))
            imwrite(CamerasRGB_box, osp.join(OutputFolder_Process, f'{idx:04d}_CamerasRGB_CropBox.png'))
            imwrite(CameraISP_box, osp.join(OutputFolder_Process, f'{idx:04d}_CameraISP_CropBox.png'))
        
        DisplaysRGB = _mean_blur_rgb(DisplaysRGB, kernel_size=blur_size)

        # Resize every stream to the alignment resolution.
        h, w = CameraRaw_ROI.shape[:2]
        resize_width = max(1, int(round(w * scale)))
        resize_height = max(1, int(round(h * scale)))
        Scale_CameraRaw = cv2.resize(CameraRaw_ROI, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        Scale_CamerasRGB = cv2.resize(CamerasRGB_ROI, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        Scale_CameraISP = cv2.resize(CameraISP_ROI, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        Scale_DisplaysRGB = cv2.resize(DisplaysRGB, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        
        # Save ROI diagnostics before optical-flow alignment.
        if save_process_img:
            imwrite(Scale_CamerasRGB, osp.join(OutputFolder_Process, f'{idx:04d}_CamerasRGB_Crop.png'))
            imwrite(Scale_CameraISP, osp.join(OutputFolder_Process, f'{idx:04d}_CameraISP_Crop.png'))
            imwrite(Scale_DisplaysRGB, osp.join(OutputFolder_Process, f'{idx:04d}_DisplaysRGB_Crop.png'))
        
        CameraRaw_ROI = torch.from_numpy(Scale_CameraRaw).float().cuda().permute(2, 0, 1).unsqueeze(0)
        CamerasRGB_ROI = torch.from_numpy(Scale_CamerasRGB).float().cuda().permute(2, 0, 1).unsqueeze(0)
        CameraISP_ROI = torch.from_numpy(Scale_CameraISP).float().cuda().permute(2, 0, 1).unsqueeze(0)
        DisplaysRGB = torch.from_numpy(Scale_DisplaysRGB).float().cuda().permute(2, 0, 1).unsqueeze(0)
        
        Hs = [CameraRaw_ROI.shape[-2], CamerasRGB_ROI.shape[-2], CameraISP_ROI.shape[-2], DisplaysRGB.shape[-2]]
        Ws = [CameraRaw_ROI.shape[-1], CamerasRGB_ROI.shape[-1], CameraISP_ROI.shape[-1], DisplaysRGB.shape[-1]]

        H_common = (min(Hs) // 16) * 16
        W_common = (min(Ws) // 16) * 16
        if H_common == 0 or W_common == 0:
            raise ValueError(f"Image too small to crop to 16-multiple. Hs={Hs}, Ws={Ws}")

        def _center_crop_to_hw_multiple_16(t: torch.Tensor, H16: int, W16: int) -> torch.Tensor:
            _, _, H, W = t.shape
            if H16 > H or W16 > W:
                raise ValueError(f"Target crop ({H16},{W16}) larger than input ({H},{W}).")
            top  = (H - H16) // 2
            left = (W - W16) // 2
            return t[..., top:top + H16, left:left + W16]

        CameraRaw_ROI   = _center_crop_to_hw_multiple_16(CameraRaw_ROI,   H_common, W_common)
        CamerasRGB_ROI  = _center_crop_to_hw_multiple_16(CamerasRGB_ROI,  H_common, W_common)
        CameraISP_ROI   = _center_crop_to_hw_multiple_16(CameraISP_ROI,   H_common, W_common)
        DisplaysRGB     = _center_crop_to_hw_multiple_16(DisplaysRGB,     H_common, W_common)
        
        with torch.no_grad():
            _, flow_up_ISP = model(CamerasRGB_ROI, CameraISP_ROI, iters=args.flow_iterations, test_mode=True)
            _, flow_up_Display = model(CamerasRGB_ROI, DisplaysRGB, iters=args.flow_iterations, test_mode=True)
            
            warped_CameraISP = warp_image(CameraISP_ROI, flow_up_ISP)
            warped_DisplaysRGB = warp_image(DisplaysRGB, flow_up_Display)
            
            out_CameraRaw = CameraRaw_ROI.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_CamerasRGB = CamerasRGB_ROI.detach().squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            out_CameraISP = warped_CameraISP.detach().squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            out_DisplaysRGB = warped_DisplaysRGB.detach().squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            del CameraRaw_ROI
            del CamerasRGB_ROI
            del CameraISP_ROI
            del DisplaysRGB
            del warped_CameraISP
            del warped_DisplaysRGB

        torch.cuda.empty_cache()

        # Crop the edge pixels of the aligned pairs
        top, left, bottom, right = cropping_border, cropping_border, resize_height-cropping_border, resize_width-cropping_border
        out_CameraRaw = out_CameraRaw[top:bottom, left:right]
        out_CamerasRGB = out_CamerasRGB[top:bottom, left:right]
        out_CameraISP = out_CameraISP[top:bottom, left:right]
        out_DisplaysRGB = out_DisplaysRGB[top:bottom, left:right]
        
        psnr_isp = calculate_psnr(out_CameraISP, out_DisplaysRGB)
        ssim_isp = calculate_ssim(out_CameraISP, out_DisplaysRGB)

        out_CameraRaw_png = (np.clip(cv2.cvtColor(out_CameraRaw.astype(np.float32), cv2.COLOR_RGB2BGR), 0, 1) * 255).astype(np.uint8)
        psnr_raw = calculate_psnr(out_CameraRaw_png, out_DisplaysRGB)
        ssim_raw = calculate_ssim(out_CameraRaw_png, out_DisplaysRGB)

        
        out_flie_Flow.write(
            f'{idx:04d} '
            f'Raw_vs_Display (PSNR: {psnr_raw:.2f}, SSIM: {ssim_raw:.3f}) | '
            f'ISP_vs_Display (PSNR: {psnr_isp:.2f}, SSIM: {ssim_isp:.3f})\n' 
        )
        out_flie_Flow.flush()
        
        np.save(osp.join(OutputFolder_Flow, f'{idx:04d}_CameraRaw.npy'), np.clip(out_CameraRaw, 0, 1).astype(np.float64))
        imwrite(out_DisplaysRGB, osp.join(OutputFolder_Flow, f'{idx:04d}_DisplaysRGB.png'))
        if save_process_img:
            imwrite(out_CameraRaw_png, osp.join(OutputFolder_Flow, f'{idx:04d}_CameraRaw.png'))
            imwrite(out_CamerasRGB, osp.join(OutputFolder_Flow, f'{idx:04d}_CamerasRGB.png'))
            imwrite(out_CameraISP, osp.join(OutputFolder_Flow, f'{idx:04d}_CameraISP.png'))
        
    out_flie_Flow.close()
    out_flie_exposure.close()

if __name__ == '__main__':
    
    main()
