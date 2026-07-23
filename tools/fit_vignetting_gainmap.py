#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate vignetting gain maps from flat-field captures.

The input may contain linear RGB flat fields (TIFF, PNG, NPY, or NPZ) or RAW
flat fields. Two maps are produced: one normalized to the image center and one
normalized to the edge. Optional Gaussian smoothing, gain clipping, ROI
restriction, and a uniformity self-test are provided.

RAW inputs are converted to linear demosaiced RGB with the local metadata and
ISP helpers. Outputs include a metadata-bearing ``<prefix>_gainmaps.npz`` and,
when ImageIO is installed, float32 TIFF maps and diagnostic previews.
"""

import argparse
import json
import os
import glob
import math
from datetime import datetime
import numpy as np

RAW_EXTS = {".dng", ".nef", ".cr2", ".cr3", ".arw", ".raf", ".rw2", ".srw", ".orf"}

# Optional third-party I/O dependencies.
try:
    import imageio.v3 as iio
except Exception:
    iio = None

try:
    import rawpy
except Exception:
    rawpy = None

# PyTorch is optional for non-RAW inputs.
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# Local RAW metadata and ISP helpers.
from metadata import get_metadata
from isp_pipeline import run_pipeline


def _ensure_rgb(arr, path):
    arr = np.asarray(arr).transpose(1,2,0)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{path}: expected an HxWx3 linear RGB image")
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _maybe_to_cuda(x):
    if not HAS_TORCH:
        return x
    try:
        return x.cuda() if torch.cuda.is_available() else x
    except Exception:
        return x


def make_rect_roi_mask(h, w, center_x, center_y, abs_h):
    """Return a bounded 16:9 rectangular ROI mask and its box metadata.

    ``abs_h`` determines the requested height. The box is translated to remain
    inside the frame and is scaled down only when the requested size cannot fit.
    Coordinates ``x2`` and ``y2`` use half-open interval semantics.
    """
    # Compute the requested 16:9 dimensions.
    roi_h = int(round(float(abs_h)))
    roi_w = int(round(roi_h * 16.0 / 9.0))

    # Scale both dimensions together if the requested ROI is too large.
    if roi_h > h or roi_w > w:
        scale = min(h / max(1.0, roi_h), w / max(1.0, roi_w))
        scale = max(0.0, min(1.0, scale))
        roi_h = int(max(1, math.floor(roi_h * scale)))
        roi_w = int(max(1, math.floor(roi_w * scale)))

    # Construct the initial box around the requested center.
    half_h = roi_h // 2
    half_w = roi_w // 2
    x1 = int(round(center_x)) - half_w
    y1 = int(round(center_y)) - half_h
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    # Translate the box into the frame without changing its size.
    if x1 < 0:
        x2 -= x1  # x2 += (-x1)
        x1 = 0
    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 -= shift
        if x1 < 0:  # Clip as a final fallback for an extremely small frame.
            x1 = 0
            x2 = min(w, roi_w)

    if y1 < 0:
        y2 -= y1
        y1 = 0
    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 -= shift
        if y1 < 0:
            y1 = 0
            y2 = min(h, roi_h)

    # Clamp coordinates defensively.
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    roi_w = max(0, x2 - x1)
    roi_h = max(0, y2 - y1)

    mask = np.zeros((h, w), dtype=bool)
    if roi_w > 0 and roi_h > 0:
        mask[y1:y2, x1:x2] = True

    return mask, (x1, y1, x2, y2, roi_w, roi_h)





def _read_linear_rgb_from_raw(path):
    if rawpy is None:
        raise RuntimeError("rawpy is required for RAW input; install `rawpy`")

    rp = rawpy.imread(path)  # rawpy.RawPy
    meta = get_metadata(rp)

    # Supply the metadata fields required by the retained ISP implementation.
    meta['alpha'] = 0.5
    meta['ref'] = 'D65'
    meta['demosaic_type'] = 'bilinear'
    meta['color_desc'] = 'RGBG'
    meta['gamma_type'] = 'Rec709'

    # Extract the visible linear mosaic.
    raw = rp.raw_image_visible.astype(np.float32)

    # Convert metadata arrays to the tensor shapes expected by the ISP.
    if HAS_TORCH:
        if 'wb_matrix' in meta:
            meta['wb_matrix'] = torch.tensor(meta['wb_matrix'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            meta['wb_matrix'] = _maybe_to_cuda(meta['wb_matrix'])
        if 'color_mask' in meta:
            cm = meta['color_mask']
            if not isinstance(cm, np.ndarray):
                cm = np.asarray(cm)
            meta['color_mask'] = torch.from_numpy(cm).unsqueeze(0).unsqueeze(0)
            meta['color_mask'] = _maybe_to_cuda(meta['color_mask'])
        if 'rgb_xyz_matrix' in meta:
            meta['rgb_xyz_matrix'] = torch.tensor(meta['rgb_xyz_matrix'], dtype=torch.float32).unsqueeze(0)
            meta['rgb_xyz_matrix'] = _maybe_to_cuda(meta['rgb_xyz_matrix'])

    # Run the RAW-to-linear-demosaiced-RGB portion of the ISP.
    meta['wb_matrix'] = torch.ones((1, 1, 1, 4), dtype=torch.float32).cuda()
    out = run_pipeline(torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).cuda(), meta, 'raw', 'demosaic')

    # Convert the result to an HxWx3 NumPy array.
    if HAS_TORCH and torch.is_tensor(out):
        out = out.squeeze(0).detach().cpu().numpy()
    else:
        out = np.asarray(out)
        if out.ndim >= 4:
            out = np.squeeze(out, axis=(0, 1))
    out = _ensure_rgb(out, path)

    # Preserve sensor orientation; callers may apply orientation separately.
    return out


def load_stack(folder):
    files = sorted(
        f for f in glob.glob(os.path.join(folder, "*"))
        if os.path.splitext(f)[1].lower() in RAW_EXTS
        or os.path.splitext(f)[1].lower() in [".tif", ".tiff", ".png", ".npy", ".npz"]
    )
    if not files:
        raise FileNotFoundError(f"No supported flat-field files found in {folder}")
    imgs = []
    shape0 = None
    for fp in files:
        arr = _read_linear_rgb_from_raw(fp)
        if shape0 is None:
            shape0 = arr.shape
        if arr.shape != shape0:
            raise ValueError(
                f"Shape mismatch: {fp} has {arr.shape}, expected {shape0}"
            )
        imgs.append(arr)
    stack = np.stack(imgs, axis=0)  # [N,H,W,3]
    return stack, files


def compute_master_flat(stack, eps=1e-8):
    master = np.mean(stack, axis=0).astype(np.float32)
    return np.maximum(master, eps)


def gaussian_smooth(img, sigma=0):
    if sigma is None or sigma <= 0:
        return img
    try:
        import cv2
        k = int(max(3, (sigma // 2) * 2 + 1))  # odd
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[..., c] = cv2.GaussianBlur(
                img[..., c], (k, k), sigmaX=float(sigma), sigmaY=float(sigma),
                borderType=cv2.BORDER_REFLECT101
            )
        return out
    except Exception:
        return img


def make_masks(h, w, center_r=0.10, edge_r0=0.90):
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    half_diag = math.hypot(cx, cy) + 1e-8
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / half_diag
    return (r <= float(center_r)), (r >= float(edge_r0)), r


def channel_means(img, mask):
    return np.array([float(img[..., c][mask].mean()) for c in range(3)], dtype=np.float32)


def build_gain_maps(master_flat, center_r=0.10, edge_r0=0.90,
                    smooth_sigma=0, max_gain=None, eps=1e-8,
                    roi_mask=None, zero_outside=True):
    """Build center- and edge-normalized gain maps.

    When ``roi_mask`` is provided, statistics use only its intersection with
    the center and edge regions. If ``zero_outside`` is enabled, values outside
    the ROI are reset to the neutral gain of one.
    """
    h, w, _ = master_flat.shape
    base = gaussian_smooth(master_flat, sigma=smooth_sigma)
    base = np.maximum(base, eps)

    center_mask, edge_mask, rmap = make_masks(h, w, center_r=center_r, edge_r0=edge_r0)

    if roi_mask is not None:
        # Restrict the statistical regions to the ROI.
        center_mask = center_mask & roi_mask
        edge_mask = edge_mask & roi_mask
        # Fall back to the full ROI when an intersection is empty.
        if not center_mask.any():
            center_mask = roi_mask.copy()
        if not edge_mask.any():
            edge_mask = roi_mask.copy()

    center_mean = channel_means(base, center_mask)
    edge_mean   = channel_means(base, edge_mask)

    gain_center = base.copy()
    gain_edge   = base.copy()
    for c in range(3):
        gain_center[..., c] = center_mean[c] / gain_center[..., c]
        gain_edge  [..., c] = edge_mean  [c] / gain_edge  [..., c]

    if max_gain is not None and max_gain > 0:
        gain_center = np.clip(gain_center, 0.0, float(max_gain))
        gain_edge   = np.clip(gain_edge,   0.0, float(max_gain))

    # Retain fitted gains inside the ROI and use neutral gain elsewhere.
    if roi_mask is not None and zero_outside:
        keep = roi_mask[..., None]  # HxWx1, bool
        inv  = (~roi_mask)[..., None]
        gain_center = gain_center * keep + inv.astype(np.float32)  # Outside ROI = 1.
        gain_edge   = gain_edge   * keep + inv.astype(np.float32)

    meta = {
        "center_r": float(center_r),
        "edge_r0": float(edge_r0),
        "smooth_sigma": float(smooth_sigma),
        "max_gain": float(max_gain) if max_gain is not None else None,
        "roi_enabled": roi_mask is not None,
        "roi_outside_value": 1,
    }
    return gain_center.astype(np.float32), gain_edge.astype(np.float32), meta, (center_mask, edge_mask, rmap)


def _normalize_to_uint8(img, pct=99.5, eps=1e-8):
    """Scale an HxWx3 float image to uint8 for robust visualization."""
    img = np.asarray(img, dtype=np.float32)
    # Use one robust scale across all channels.
    hi = np.percentile(img, pct)
    scale = 255.0 / max(hi, eps)
    out = np.clip(img * scale, 0, 255).astype(np.uint8)
    return out


def _draw_rect_border_inplace(img, x1, y1, x2, y2, color=(255, 0, 0), thickness=2):
    """Draw an RGB rectangle in place without requiring OpenCV."""
    H, W = img.shape[:2]
    x1 = max(0, min(int(x1), W - 1))
    x2 = max(0, min(int(x2), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    y2 = max(0, min(int(y2), H - 1))
    if x2 <= x1 or y2 <= y1:
        return img

    t = max(1, int(thickness))
    # Top edge.
    img[y1:y1+t, x1:x2, :] = color
    # Bottom edge.
    img[y2-t:y2, x1:x2, :] = color
    # Left edge.
    img[y1:y2, x1:x1+t, :] = color
    # Right edge.
    img[y1:y2, x2-t:x2, :] = color
    return img


def save_roi_visuals(gain_center, gain_edge, out_dir, prefix, roi_meta, border_thickness=2):
    """Write uint8 previews with a red ROI border.

      <prefix>_gain_center1_vis_roi.tiff
      <prefix>_gain_edge1_vis_roi.tiff
    """
    if roi_meta is None:
        return None, None
    os.makedirs(out_dir, exist_ok=True)
    x1, y1, x2, y2, rw, rh = roi_meta

    # Normalize gain maps for preview.
    vis_c = _normalize_to_uint8(gain_center, pct=99.5)
    vis_e = _normalize_to_uint8(gain_edge,   pct=99.5)

    # Draw the ROI in red.
    _draw_rect_border_inplace(vis_c, x1, y1, x2, y2, color=(255, 0, 0), thickness=border_thickness)
    _draw_rect_border_inplace(vis_e, x1, y1, x2, y2, color=(255, 0, 0), thickness=border_thickness)

    # Write uint8 TIFF previews.
    path_c = os.path.join(out_dir, f"{prefix}_gain_center1_vis_roi.tiff")
    path_e = os.path.join(out_dir, f"{prefix}_gain_edge1_vis_roi.tiff")

    if iio is not None:
        iio.imwrite(path_c, vis_c, dtype=np.uint8)
        iio.imwrite(path_e, vis_e, dtype=np.uint8)
    else:
        print("[WARN] ImageIO is unavailable; skipping ROI TIFF previews")
        path_c = path_e = None

    return path_c, path_e


def save_outputs(gain_center, gain_edge, out_dir, prefix, meta, src_files):
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{prefix}_gainmaps.npz")
    meta_full = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "shape": list(gain_center.shape),
        "dtype": "float32",
        "methods": ["center=1", "edge=1"],
        "params": meta,
        "sources": src_files,
        "version": "1.1-fixed-import",
    }
    np.savez(npz_path,
             gain_center1=gain_center,
             gain_edge1=gain_edge,
             meta_json=json.dumps(meta_full, ensure_ascii=False))
    t_center = t_edge = None
    if iio is not None:
        t_center = os.path.join(out_dir, f"{prefix}_gain_center1.tiff")
        t_edge = os.path.join(out_dir, f"{prefix}_gain_edge1.tiff")
        iio.imwrite(t_center, gain_center, dtype=np.float32)
        iio.imwrite(t_edge, gain_edge, dtype=np.float32)
    else:
        print("[WARN] ImageIO is unavailable; skipping TIFF gain maps")
    return npz_path, t_center, t_edge


def _uniformity_report(img, center_mask, edge_mask):
    rep = {}
    rep["shape"] = list(img.shape)
    for name, mask in [("center", center_mask), ("edge", edge_mask), ("all", np.ones(img.shape[:2], bool))]:
        vals = img[mask].reshape(-1, img.shape[2])  # Nx3
        mean = vals.mean(axis=0); std = vals.std(axis=0)
        vmin = vals.min(axis=0); vmax = vals.max(axis=0)
        rep[f"{name}_mean"] = mean.tolist()
        rep[f"{name}_std"] = std.tolist()
        rep[f"{name}_min"] = vmin.tolist()
        rep[f"{name}_max"] = vmax.tolist()
        rep[f"{name}_cv%"] = (std / (mean + 1e-8) * 100.0).tolist()
    return rep


def _save_self_test_outputs(out_dir, prefix, master_flat, gain, center_mask, edge_mask):
    os.makedirs(out_dir, exist_ok=True)
    corrected = master_flat * gain
    mean_all = corrected.mean(axis=(0, 1), keepdims=True)
    rel = corrected / (mean_all + 1e-8)  # Values closer to one are more uniform.
    rel_gray = rel.mean(axis=2)

    report = {
        "before": _uniformity_report(master_flat, center_mask, edge_mask),
        "after": _uniformity_report(corrected, center_mask, edge_mask),
        "note": "float32 linear domain; rel_gray=corrected relative luminance (ideal white=1)"
    }
    with open(os.path.join(out_dir, f"{prefix}_selftest_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if iio is not None:
        iio.imwrite(os.path.join(out_dir, f"{prefix}_selftest_corrected.tiff"),
                    corrected.astype(np.float32))
        lo, hi = 0.97, 1.03
        heat = np.clip((rel_gray - lo) / (hi - lo), 0, 1).astype(np.float32)
        iio.imwrite(os.path.join(out_dir, f"{prefix}_selftest_rel_gray.tiff"), heat)
    else:
        print("[WARN] ImageIO is unavailable; skipping self-test TIFF files")

    af = report["after"]
    import numpy as _np
    cv = _np.array(af["all_cv%"])
    print("[Self-Test] corrected full-frame CV% (R,G,B):", _np.round(cv, 3))
    print("[Self-Test] corrected center/edge means (RGB):")
    print("  center_mean:", _np.round(_np.array(af["center_mean"]), 2))
    print("  edge_mean  :", _np.round(_np.array(af["edge_mean"]), 2))




def parse_args():
    ap = argparse.ArgumentParser(
        description="Build gain maps from linear RGB or RAW flat fields"
    )
    ap.add_argument("--in", dest="in_dir", required=True, help="Flat-field input directory")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output directory")
    ap.add_argument("--prefix", dest="prefix", default="gain", help="Output filename prefix")
    ap.add_argument("--center-r", type=float, default=0.10, help="Center radius normalized by the half diagonal")
    ap.add_argument("--edge-r0", type=float, default=0.90, help="Inner edge-ring radius normalized by the half diagonal")
    ap.add_argument("--smooth-sigma", type=float, default=9.0, help="Gaussian smoothing sigma in pixels; 0 disables")
    ap.add_argument("--max-gain", type=float, default=-1, help="Maximum gain; values <= 0 disable clipping")
    ap.add_argument(
        "--self-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a uniformity self-test and export its report and heat map",
    )

    ap.add_argument("--roi-center-x", type=int, default=2065, help="ROI center x in source-image pixels")
    ap.add_argument("--roi-center-y", type=int, default=1454, help="ROI center y in source-image pixels")
    ap.add_argument("--roi-abs-h", type=float, default=1032, help="Absolute ROI height in pixels")
    ap.add_argument(
        "--roi-zero-outside",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use neutral gain outside the ROI (enabled by default)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    stack, files = load_stack(args.in_dir)
    master = compute_master_flat(stack)
    H, W, _ = master.shape

    # Enable the ROI when all geometry parameters are valid.
    roi_mask = None
    roi_meta = None
    if args.roi_center_x >= 0 and args.roi_center_y >= 0 and args.roi_abs_h and args.roi_abs_h > 0:
        roi_mask, roi_meta = make_rect_roi_mask(
            H, W,
            center_x=args.roi_center_x,
            center_y=args.roi_center_y,
            abs_h=args.roi_abs_h
        )
        # Report the resolved ROI for reproducibility.
        x1, y1, x2, y2, rw, rh = roi_meta
        print(f"[ROI] rect: ({x1},{y1})-({x2},{y2}) size=({rh},{rw})")

    gain_c, gain_e, meta, masks = build_gain_maps(
        master,
        center_r=args.center_r,
        edge_r0=args.edge_r0,
        smooth_sigma=args.smooth_sigma,
        max_gain=None if args.max_gain is None or args.max_gain <= 0 else args.max_gain,
        roi_mask=roi_mask,
        zero_outside=bool(args.roi_zero_outside)
    )

    # Record resolved ROI metadata in the output archive.
    if roi_meta is not None:
        x1, y1, x2, y2, rw, rh = roi_meta
        meta["roi_rect_xyxy"] = [int(x1), int(y1), int(x2), int(y2)]
        meta["roi_wh"] = [int(rw), int(rh)]

    npz_path, t1, t2 = save_outputs(gain_c, gain_e, args.out_dir, args.prefix, meta, files)

    print("[OK] Gain maps saved:")
    print("  NPZ :", npz_path)
    if t1: print("  TIFF center=1:", t1)
    if t2: print("  TIFF edge=1  :", t2)

    if args.self_test:
        center_mask, edge_mask, _ = masks
        _save_self_test_outputs(args.out_dir, args.prefix, master, gain_c, center_mask, edge_mask)

    # Save separate ROI previews without modifying the gain-map arrays.
    vis_c, vis_e = save_roi_visuals(gain_c, gain_e, args.out_dir, args.prefix, roi_meta, border_thickness=2)
    if vis_c or vis_e:
        print("  VIS center=1 (with ROI):", vis_c)
        print("  VIS edge=1   (with ROI):", vis_e)

if __name__ == "__main__":
    main()
