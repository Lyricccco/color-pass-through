# -----------------------------------------------------------------------------------
# [ECCV2024] DualDn: Dual-domain Denoising via Differentiable ISP 
# [Homepage] https://openimaginglab.github.io/DualDn/
# [Author] Originally Written by Ruikang Li, from MMLab, CUHK.
# [License] Absolutely open-source and free to use, please cite our paper if possible. :)
# -----------------------------------------------------------------------------------

import torch
import kornia
import numpy as np

def normalize(image, color_mask, black_level, white_level):

    black_level_mask = torch.zeros(image.shape).to(image.device)
    white_level_mask = float(white_level)

    for i in range(len(black_level)):
        if torch.any(torch.eq(color_mask, i)): 
            black_level_mask = black_level_mask.masked_fill(torch.eq(color_mask, i), value = black_level[i])

    nomalize_image = image - black_level_mask
    nomalize_image = torch.clamp(nomalize_image, min=1e-8)
    nomalize_image = torch.div(nomalize_image, white_level_mask - black_level_mask)

    return nomalize_image


def white_balance(image, color_mask, wb_matrix):

    wb_mask = torch.ones(image.shape).to(image.device)
    wb_matrix = torch.div(wb_matrix, torch.min(wb_matrix, -1).values.unsqueeze(-1))

    for j in range(wb_matrix.shape[0]):
        for i in range(wb_matrix.shape[-1]):
            if torch.any(torch.eq(color_mask, i)): 
                wb_mask[j] = wb_mask[j].masked_fill(torch.eq(color_mask[j], i), value = wb_matrix[j][0][0][i])

    wb_image = torch.mul(image, wb_mask)
    wb_image = torch.clamp(wb_image, min=1e-8, max=1.0)

    return wb_image


def demosaic(image, color_desc, color_mask, rgb_xyz_matrix, demosaic_type='bilinear'):

    mask_r = torch.zeros(image.shape).to(image.device)
    mask_g = torch.zeros(image.shape).to(image.device)
    mask_b = torch.zeros(image.shape).to(image.device)

    for i, col in enumerate(color_desc):
        if col == 'R':
            mask_r = mask_r.masked_fill(torch.eq(color_mask, i), 1)
            image_r = torch.mul(image, mask_r)
        if col == 'G':
            mask_g = mask_g.masked_fill(torch.eq(color_mask, i), 1)
            image_g = torch.mul(image, mask_g)
        if col == 'B':
            mask_b = mask_b.masked_fill(torch.eq(color_mask, i), 1)
            image_b = torch.mul(image, mask_b)

    if demosaic_type == 'nearest':
        image_r = image_r + torch.roll(image_r, shifts=1, dims=-2) + torch.roll(image_r, shifts=1, dims=-1) \
        + torch.roll(image_r, shifts=(1, 1), dims=(-1, -2))
        image_g = image_g + torch.roll(image_g, shifts=1, dims=-1)
        image_b = image_b + torch.roll(image_b, shifts=1, dims=-2) + torch.roll(image_b, shifts=1, dims=-1) \
        + torch.roll(image_b, shifts=(1, 1), dims=(-1, -2))

    elif demosaic_type == 'bilinear':
        r_b_kernel1 = torch.tensor([[1, 0, 1],
                                    [0, 0, 0],
                                    [1, 0, 1]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4.0
        r_b_kernel2 = torch.tensor([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 2.0
        g_kernel = torch.tensor([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4.0
        image_r = image_r + torch.nn.functional.conv2d(image_r, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_r, r_b_kernel2, stride=(1, 1), padding="same")
        image_g = image_g + torch.nn.functional.conv2d(image_g, g_kernel, stride=(1, 1), padding="same")
        image_b = image_b + torch.nn.functional.conv2d(image_b, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_b, r_b_kernel2, stride=(1, 1), padding="same")

    elif demosaic_type == 'Malvar':
        f0 = torch.tensor([[0, 0, -1, 0, 0],
                        [0, 0, 2, 0, 0],
                        [-1, 2, 4, 2, -1],
                        [0, 0, 2, 0, 0],
                        [0, 0, -1, 0, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 8

        f1 = torch.tensor(
            [[0, 0, 1, 0, 0],
            [0, -2, 0, -2, 0],
            [-2, 8, 10, 8, -2],
            [0, -2, 0, -2, 0],
            [0, 0, 1, 0, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 16

        f2 = torch.transpose(f1, dim0=-2, dim1=-1).to(image.device)

        f3 = torch.tensor(
            [[0, 0, -3, 0, 0],
            [0, 4, 0, 4, 0],
            [-3, 0, 12, 0, -3],
            [0, 4, 0, 4, 0],
            [0, 0, -3, 0, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 16

        d0 = torch.nn.functional.conv2d(image, f0, stride=1, padding="same")
        d1 = torch.nn.functional.conv2d(image, f1, stride=1, padding="same")
        d2 = torch.nn.functional.conv2d(image, f2, stride=1, padding="same")
        d3 = torch.nn.functional.conv2d(image, f3, stride=1, padding="same")

        mask_r_g_r_row = torch.roll(mask_r, shifts=1, dims=-1)
        mask_r_g_r_col = torch.roll(mask_r, shifts=1, dims=-2)
        image_r = image_r + torch.mul(d1, mask_r_g_r_row) + torch.mul(d2, mask_r_g_r_col) + torch.mul(d3, mask_b)

        image_g = image_g + torch.mul(d0, mask_r) + torch.mul(d0, mask_b)

        mask_b_g_b_row = torch.roll(mask_b, shifts=1, dims=-1)
        mask_b_g_b_col = torch.roll(mask_b, shifts=1, dims=-2)
        image_b = image_b + torch.mul(d1, mask_b_g_b_row) + torch.mul(d2, mask_b_g_b_col) + torch.mul(d3, mask_r)

    elif demosaic_type == 'AHD':
        h0_row = torch.tensor([-1, 2, 2, 2, -1], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) / 4

        h0_col = torch.tensor([[-1],
                        [2],
                        [2],
                        [2],
                        [-1]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4

        d0_row = torch.nn.functional.conv2d(image, h0_row, stride=1, padding="same")
        d0_col = torch.nn.functional.conv2d(image, h0_col, stride=1, padding="same")

        g_row = image_g + torch.mul(d0_row, mask_r) + torch.mul(d0_row, mask_b)
        g_col = image_g + torch.mul(d0_col, mask_r) + torch.mul(d0_col, mask_b)

        r_b_kernel1 = torch.tensor([[1, 0, 1],
                                    [0, 0, 0],
                                    [1, 0, 1]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4.0
        r_b_kernel2 = torch.tensor([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 2.0
        g_kernel = torch.tensor([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4.0

        homo_row1 = torch.tensor([1, 2, -3, 0, 0], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
        homo_col1 = torch.tensor([[1],
                        [2],
                        [-3],
                        [0],
                        [0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) 

        homo_row2 = torch.tensor([0, 0, -3, 2, 1], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
        homo_col2 = torch.tensor([[0],
                        [0],
                        [-3],
                        [2],
                        [1]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) 

        r_sample = image_r + torch.nn.functional.conv2d(image_r, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_r, r_b_kernel2, stride=(1, 1), padding="same")
        g_sample = image_g + torch.nn.functional.conv2d(image_g, g_kernel, stride=(1, 1), padding="same")
        b_sample = image_b + torch.nn.functional.conv2d(image_b, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_b, r_b_kernel2, stride=(1, 1), padding="same")

        r_row = kornia.filters.gaussian_blur2d((r_sample - g_sample), (3,3), (1.5,1.5)) + g_row
        r_col = kornia.filters.gaussian_blur2d((r_sample - g_sample), (3,3), (1.5,1.5)) + g_col

        b_row = kornia.filters.gaussian_blur2d((b_sample - g_sample), (3,3), (1.5,1.5)) + g_row
        b_col = kornia.filters.gaussian_blur2d((b_sample - g_sample), (3,3), (1.5,1.5)) + g_col

        image_sample = torch.concat([r_sample, g_sample, b_sample], dim=1)

        image_xyz = torch.matmul(rgb_xyz_matrix.unsqueeze(-3), image_sample.permute(0,2,3,1).unsqueeze(-1)).squeeze(-1).permute(0,3,1,2)
        xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)[..., :, None, None].to(image.device).unsqueeze(0)
        xyz_normalized = torch.div(image_xyz, xyz_ref_white)

        threshold = 0.008856
        power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
        scale = 7.787 * xyz_normalized + 4.0 / 29.0
        xyz_int = torch.where(xyz_normalized > threshold, power, scale)

        L = ((116.0 * xyz_int[..., 1, :, :]) - 16.0).unsqueeze(-3)
        a = (500.0 * (xyz_int[..., 0, :, :] - xyz_int[..., 1, :, :])).unsqueeze(-3)
        b = (200.0 * (xyz_int[..., 1, :, :] - xyz_int[..., 2, :, :])).unsqueeze(-3)

        row1 = torch.abs(torch.nn.functional.conv2d(L, homo_row1, stride=(1, 1), padding="same") ) + torch.sqrt(torch.square(torch.nn.functional.conv2d(a, homo_row1, stride=(1, 1), padding="same")) + torch.square(torch.nn.functional.conv2d(b, homo_row1, stride=(1, 1), padding="same")))
        col1 = torch.abs(torch.nn.functional.conv2d(L, homo_col1, stride=(1, 1), padding="same") ) + torch.sqrt(torch.square(torch.nn.functional.conv2d(a, homo_col1, stride=(1, 1), padding="same")) + torch.square(torch.nn.functional.conv2d(b, homo_col1, stride=(1, 1), padding="same")))

        row2 = torch.abs(torch.nn.functional.conv2d(L, homo_row2, stride=(1, 1), padding="same") ) + torch.sqrt(torch.square(torch.nn.functional.conv2d(a, homo_row2, stride=(1, 1), padding="same")) + torch.square(torch.nn.functional.conv2d(b, homo_row2, stride=(1, 1), padding="same")))
        col2 = torch.abs(torch.nn.functional.conv2d(L, homo_col2, stride=(1, 1), padding="same") ) + torch.sqrt(torch.square(torch.nn.functional.conv2d(a, homo_col2, stride=(1, 1), padding="same")) + torch.square(torch.nn.functional.conv2d(b, homo_col2, stride=(1, 1), padding="same")))

        row = row1 + row2
        col = col1 + col2

        image_r = torch.where(row >= col, r_col, r_row)
        image_g = torch.where(row >= col, g_col, g_row)
        image_b = torch.where(row >= col, b_col, b_row)

    demosaic_image = torch.concat([image_r, image_g, image_b], dim=1)

    return demosaic_image


def rgb_to_srgb(image, rgb_xyz_matrix, ref='D65'):

    if ref == 'D65':
        xyz_srgb_matrix = torch.tensor([[3.2404542, -1.5371385, -0.4985314],
                                        [-0.9692660, 1.8760108, 0.0415560],
                                        [0.0556434, -0.2040259, 1.0572252]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image.device)
    elif ref == 'D50':
        xyz_srgb_matrix = torch.tensor([[3.1338561, -1.6168667, -0.4906146],
                                        [-0.9787684, 1.9161415, 0.0334540],
                                        [0.0719453, -0.2289914, 1.4052427]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image.device)
    elif ref == 'same':
        xyz_srgb_matrix = torch.tensor([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image.device)

    xyz_srgb_matrix = xyz_srgb_matrix / torch.sum(xyz_srgb_matrix, axis=-1, keepdims=True)

    xyz_image = torch.matmul(rgb_xyz_matrix.unsqueeze(-3), image.permute(0,2,3,1).unsqueeze(-1)).squeeze(-1).permute(0,3,1,2)
    srgb_image = torch.matmul(xyz_srgb_matrix.unsqueeze(-3), xyz_image.permute(0,2,3,1).unsqueeze(-1)).squeeze(-1).permute(0,3,1,2)

    return srgb_image


def gamma(image, gamma_type='Rec709'):

    if gamma_type == 'Rec709':
        gamma_image1 = 4.5*image.masked_fill(image>=0.018, 0)
        mask = torch.zeros(image.shape).to(image.device)
        gamma_image2 = 1.099*torch.pow(image.masked_fill(image<0.018, 0), 0.45) + mask.masked_fill(image>=0.018, -0.099)
        gamma_image = gamma_image1 + gamma_image2

    if gamma_type == '2.2':
        gamma_image = torch.pow(torch.clamp(image, min=1e-8), 1/2.2)

    return gamma_image


def tone_mapping(image, alpha):

    mid_image = image + alpha*(image)*(1-image)
    tone_mapping_image = mid_image + alpha*(mid_image)*(1-mid_image)

    return tone_mapping_image


def run_pipeline(image, metadata, input_stage, output_stage):

##* special case
    # if metadata['raw_type'] == 'Stack':
    ##! TODO Process stack RAW

##*1  raw stage
    current_image = image
    current_stage = 'raw'
    if current_stage == output_stage:
        return current_image


##*2  normalize stage 
    if input_stage == current_stage:
        current_image = normalize(current_image, metadata['color_mask'], metadata['black_level'], metadata['white_level'])
        input_stage = 'normal'
    current_stage = 'normal'
    if current_stage == output_stage:
        return current_image


##*3  white_balance stage
    if input_stage == current_stage:
        # when training with normalized torch.tensors
        if type(current_image) is np.ndarray: 
            current_image = torch.from_numpy(current_image.astype(np.float32)).to('cuda').unsqueeze(0).unsqueeze(0)
        current_image = white_balance(current_image, metadata['color_mask'], metadata['wb_matrix'])
        input_stage = 'white_balance'
    current_stage = 'white_balance'
    if current_stage == output_stage: 
        return current_image


##*4  demosaic stage
    if input_stage == current_stage:
        current_image = demosaic(current_image, metadata['color_desc'], metadata['color_mask'], metadata['rgb_xyz_matrix'], metadata['demosaic_type'])
        input_stage = 'demosaic'
    current_stage = 'demosaic'
    if current_stage == output_stage:
        return current_image


##*5  rgb_to_srgb stage
    if input_stage == current_stage:
        current_image = rgb_to_srgb(current_image, metadata['rgb_xyz_matrix'], metadata['ref'])
        input_stage = 'rgb_to_srgb'
    current_stage = 'rgb_to_srgb'
    if current_stage == output_stage:
        return current_image


##*6  gamma stage
    if input_stage == current_stage:
        current_image = gamma(current_image, metadata['gamma_type'])
        input_stage = 'gamma'
    current_stage = 'gamma'
    if current_stage == output_stage:
        return current_image


##*7 tone_mapping stage
    if input_stage == current_stage:
        current_image = tone_mapping(current_image, metadata['alpha'])
        input_stage = 'tone_mapping'
    current_stage = 'tone_mapping'
    if current_stage == output_stage:
        return current_image

    #! invalid input/output stage!
    raise ValueError('Invalid input/output stage: input_stage = {}, output_stage = {}'.format(input_stage, output_stage))