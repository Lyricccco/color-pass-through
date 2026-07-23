import torch
import torch.nn as nn
import torch.nn.functional as F

def _to_gray(x):
    if x.size(1) == 1: return x
    w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(1,3,1,1)
    return (x * w).sum(1, keepdim=True)

def _sobel(x):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device) / 8.0
    ky = kx.t()
    kx = kx.view(1,1,3,3).repeat(x.size(1),1,1,1)
    ky = ky.view(1,1,3,3).repeat(x.size(1),1,1,1)
    gx = F.conv2d(x, kx, padding=1, groups=x.size(1))
    gy = F.conv2d(x, ky, padding=1, groups=x.size(1))
    return gx, gy

def _percentile_per_batch(x, q):
    flat = x.reshape(x.shape[0], -1)
    t = torch.quantile(flat, torch.tensor(q, device=x.device, dtype=x.dtype), dim=1)
    return t.view(-1,1,1,1)

def _dilate01(mask01, r):
    if r <= 0: return mask01
    k = 2*r+1
    return F.max_pool2d(mask01, kernel_size=k, stride=1, padding=r)

class EdgeMismatchMask(nn.Module):
    """Detect mismatches by comparing edge neighborhoods.

    An edge in one image is marked as mismatched when no corresponding edge is
    found within radius ``r`` in the other image. Symmetric comparison is
    optional.
    """
    def __init__(self, edge_q=0.80, radius_px=2, band_px=1, symmetric=True):
        super().__init__()
        self.edge_q = edge_q         # Edge-strength quantile threshold.
        self.radius_px = radius_px   # Edge-matching tolerance radius.
        self.band_px = band_px       # Optional mask dilation width.
        self.symmetric = symmetric

    @torch.no_grad()
    def forward(self, I_ref, I_warped):
        # I_ref is the unwarped BxCxHxW reference image.
        # I_warped is a BxCxHxW image warped into the reference coordinates.
        g1 = _to_gray(I_ref)
        g2 = _to_gray(I_warped)

        gx1, gy1 = _sobel(g1); mag1 = torch.sqrt(gx1**2 + gy1**2 + 1e-12)
        gx2, gy2 = _sobel(g2); mag2 = torch.sqrt(gx2**2 + gy2**2 + 1e-12)

        t1 = _percentile_per_batch(mag1, self.edge_q)
        t2 = _percentile_per_batch(mag2, self.edge_q)
        E1 = (mag1 > t1).float()     # Reference edges.
        E2 = (mag2 > t2).float()     # Warped-image edges.

        # Morphological dilation supplies the matching tolerance.
        E1d = _dilate01(E1, self.radius_px)
        E2d = _dilate01(E2, self.radius_px)

        mis_12 = E2 * (1 - E1d)  # Edges present only in the warped image.
        if self.symmetric:
            mis_21 = E1 * (1 - E2d)  # Edges missing from the warped image.
            mask = torch.clamp(mis_12 + mis_21, 0, 1)
        else:
            mask = mis_12

        if self.band_px > 0:
            mask = _dilate01(mask, self.band_px)

        return mask  # Bx1xHxW, {0,1}
