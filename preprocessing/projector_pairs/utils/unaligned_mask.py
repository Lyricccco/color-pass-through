import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- basic ops ----
def _to_gray(x):
    # x: BxCxHxW in [0,1] or [0,255]; returns Bx1xHxW.
    if x.size(1) == 1:
        g = x
    else:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(1,3,1,1)
        g = (x * w).sum(1, keepdim=True)
    return g

def _sobel_conv(x):
    # x: BxCxHxW -> returns grad_x, grad_y (same shape)
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device) / 8.0
    ky = kx.t()
    kx = kx.view(1,1,3,3); ky = ky.view(1,1,3,3)
    C = x.size(1)
    kx = kx.repeat(C,1,1,1); ky = ky.repeat(C,1,1,1)
    pad = (1,1,1,1)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return gx, gy

def _percentile_per_batch(x, q):
    # x: Bx1xHxW or BxHxW ; q in [0,1]
    B = x.shape[0]
    flat = x.reshape(B, -1)
    t = torch.quantile(flat, torch.tensor(q, device=x.device, dtype=x.dtype), dim=1)
    return t.view(B, 1, 1, 1)

def _dilate(mask01, r):
    # Dilate a Bx1xHxW binary mask with max pooling.
    if r <= 0: return mask01
    k = 2*r + 1
    return F.max_pool2d(mask01, kernel_size=k, stride=1, padding=r)

def _erode(mask01, r):
    if r <= 0: return mask01
    k = 2*r + 1
    return 1. - F.max_pool2d(1. - mask01, kernel_size=k, stride=1, padding=r)

def _close(mask01, r):
    return _erode(_dilate(mask01, r), r)

# ---- warping helpers (only needed if flows are provided) ----
def _make_base_grid(B, H, W, device, dtype):
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing='ij'
    )
    grid = torch.stack([xs, ys], dim=-1).view(1, H, W, 2).repeat(B,1,1,1)  # BxHxWx2
    return grid

def _flow_to_norm(flow):
    # Convert Bx2xHxW pixel flow to normalized grid offsets.
    B, _, H, W = flow.shape
    fx = flow[:,0:1]; fy = flow[:,1:2]
    nx = fx / ((W - 1) / 2.0)
    ny = fy / ((H - 1) / 2.0)
    return torch.cat([nx, ny], dim=1)  # Bx2xHxW

def warp_image_by_flow(img, flow):
    # img: BxCxHxW ; flow: Bx2xHxW (pixels) mapping I1->I2
    B, C, H, W = img.shape
    base = _make_base_grid(B,H,W,img.device,img.dtype)                # BxHxWx2
    delta = _flow_to_norm(flow).permute(0,2,3,1)                      # BxHxWx2
    grid = base + delta
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)

def warp_flow_by_flow(flow_to_warp, ref_flow):
    # Warp flow from image-2 coordinates into image-1 coordinates.
    B, _, H, W = flow_to_warp.shape
    base = _make_base_grid(B,H,W,flow_to_warp.device,flow_to_warp.dtype)
    delta = _flow_to_norm(ref_flow).permute(0,2,3,1)
    grid = base + delta
    fx = F.grid_sample(flow_to_warp[:,0:1], grid, mode='bilinear', padding_mode='border', align_corners=True)
    fy = F.grid_sample(flow_to_warp[:,1:2], grid, mode='bilinear', padding_mode='border', align_corners=True)
    return torch.cat([fx, fy], dim=1)

# ---- main module ----
class MisalignmentMask(nn.Module):
    def __init__(self,
                 photometric_q=0.90,       # Gradient-residual quantile.
                 edge_q=0.80,              # Edge-strength quantile.
                 edge_radius_px=2,         # Edge-alignment tolerance in pixels.
                 flow_grad_q=0.95,         # Flow-gradient quantile.
                 fb_alpha=0.01, fb_beta=0.5,  # Forward-backward consistency.
                 post_dilate=2, post_close=2  # Morphological cleanup.
                 ):
        super().__init__()
        self.photometric_q = photometric_q
        self.edge_q = edge_q
        self.edge_radius_px = edge_radius_px
        self.flow_grad_q = flow_grad_q
        self.fb_alpha = fb_alpha
        self.fb_beta = fb_beta
        self.post_dilate = post_dilate
        self.post_close = post_close

    @torch.no_grad()
    def forward(self, I1, I2_aligned, flow12=None, flow21=None):
        """
        I1 and I2_aligned are Bx3xHxW or Bx1xHxW in the same coordinates.
        Optional flow12 and flow21 are Bx2xHxW pixel displacements.
        """
        I1 = I1.contiguous()
        I2 = I2_aligned.contiguous()
        B, _, H, W = I1.shape

        # 1. Gradient residual, which is relatively robust to illumination.
        gx1, gy1 = _sobel_conv(I1)
        gx2, gy2 = _sobel_conv(I2)
        grad_res = (gx1 - gx2).abs().mean(1, keepdim=True) + (gy1 - gy2).abs().mean(1, keepdim=True)
        t_ph = _percentile_per_batch(grad_res, self.photometric_q)
        photo_mask = (grad_res > t_ph).float()  # Bx1xHxW

        # 2. Edge mismatch using grayscale Sobel magnitudes.
        g1 = _to_gray(I1)
        g2 = _to_gray(I2)
        gx1g, gy1g = _sobel_conv(g1)
        gx2g, gy2g = _sobel_conv(g2)
        mag1 = torch.sqrt(gx1g**2 + gy1g**2 + 1e-12)
        mag2 = torch.sqrt(gx2g**2 + gy2g**2 + 1e-12)
        t_e1 = _percentile_per_batch(mag1, self.edge_q)
        t_e2 = _percentile_per_batch(mag2, self.edge_q)
        E1 = (mag1 > t_e1).float()   # Bx1xHxW
        E2 = (mag2 > t_e2).float()

        # Dilation allows an edge-matching tolerance of r pixels.
        E1_d = _dilate(E1, self.edge_radius_px)
        E2_d = _dilate(E2, self.edge_radius_px)
        edge_mis_1 = (E2 * (1 - E1_d)).float()
        edge_mis_2 = (E1 * (1 - E2_d)).float()  # Symmetric mismatch.
        edge_mask = torch.clamp(edge_mis_1 + edge_mis_2, 0, 1)

        # 3. Optional flow evidence: consistency and flow gradients.
        fb_mask = torch.zeros_like(photo_mask)
        flowg_mask = torch.zeros_like(photo_mask)

        if (flow12 is not None) and (flow21 is not None):
            f21_w = warp_flow_by_flow(flow21, flow12)  # Warp into I1.
            fb_err = (flow12 + f21_w).pow(2).sum(1, keepdim=True)     # ||F12 + F21^w||^2
            mag = (flow12.pow(2) + f21_w.pow(2)).sum(1, keepdim=True) # ||F12||^2 + ||F21^w||^2
            fb_mask = (fb_err > (self.fb_alpha * mag + self.fb_beta)).float()

            # Large flow gradients often indicate motion boundaries or instability.
            ux, uy = _sobel_conv(flow12[:,0:1])
            vx, vy = _sobel_conv(flow12[:,1:1+1])
            flow_grad = (ux.abs() + uy.abs() + vx.abs() + vy.abs())
            t_fg = _percentile_per_batch(flow_grad, self.flow_grad_q)
            flowg_mask = (flow_grad > t_fg).float()

        # 4. Merge evidence and clean the mask morphologically.
        bad = torch.clamp(photo_mask + edge_mask + fb_mask + flowg_mask, 0, 1)
        if self.post_dilate > 0:
            bad = _dilate(bad, self.post_dilate)
        if self.post_close > 0:
            bad = _close(bad, self.post_close)

        stats = {
            "t_ph": t_ph.flatten().tolist(),
            "t_e1": t_e1.flatten().tolist(),
            "t_e2": t_e2.flatten().tolist()
        }
        if (flow12 is not None) and (flow21 is not None):
            stats["fb_alpha"] = self.fb_alpha
            stats["fb_beta"]  = self.fb_beta
            stats["flow_grad_q"] = self.flow_grad_q
        return bad, stats
