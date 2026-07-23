import os
import cv2
import math
import torch
import numpy as np

import torch.nn.functional as F
from torchvision.utils import make_grid

def normalize(img, maxvalue):
    min_val = np.min(img)
    max_val = np.max(img)

# Normalizing the array to the range 0-255
    normalized_result = maxvalue * (img - min_val) / (max_val - min_val)

    return normalized_result

def fix_orientation(image, orientation):
    # 1 = Horizontal (normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]
    
    # np.array [H,W]
    # torch.Tensor [H,W,C]

    if isinstance(image, np.ndarray):
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
    
    elif isinstance(image, torch.Tensor):
        if orientation == 1:
            pass
        elif orientation == 2:
            image = image.flip(0)  
        elif orientation == 3:
            image = image.rot90(2, [0, 1])   
        elif orientation == 4:
            image = image.flip(1) 
        elif orientation == 5:
            image = image.flip(0) 
            image = image.rot90(1, [0, 1])  
        elif orientation == 6:
            image = image.rot90(3, [0, 1])  
        elif orientation == 7:
            image = image.flip(0) 
            image = image.rot90(3, [0, 1]) 
        elif orientation == 8:
            image = image.rot90(1, [0, 1]) 

    else:
        raise TypeError(f"Unsupported image type for Flipping: {type(image)}")

    return image


def warp_image(image, flow):
    """
    Warps 'image' according to 'flow' using F.grid_sample.
    image: [1, C, H, W] tensor
    flow:  [1, 2, H, W] tensor with displacements in pixel units
    """
    _, _, H, W = image.shape

    # Create normalized coordinate grid [-1, 1] for height and width
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W)) # xx, yy have shape: [H, W]
    yy = yy.cuda().float()
    xx = xx.cuda().float()

    # Normalize to [-1, 1]
    grid_x = 2.0 * xx / (W - 1) - 1.0
    grid_y = 2.0 * yy / (H - 1) - 1.0

    # Add flow. But first, convert the flow from pixel units to normalized units
    flow_u = flow[:, 0, :, :]  # horizontal displacement
    flow_v = flow[:, 1, :, :]  # vertical displacement
    flow_u_norm = 2.0 * flow_u / (W - 1)
    flow_v_norm = 2.0 * flow_v / (H - 1)

    # Final sampling grid = base grid + flow
    grid_x = grid_x.unsqueeze(0) + flow_u_norm
    grid_y = grid_y.unsqueeze(0) + flow_v_norm

    # Stack them into shape [N, H, W, 2]
    grid = torch.stack((grid_x, grid_y), dim=3)

    # Warp the image using bilinear interpolation
    warped_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)

    return warped_image

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result