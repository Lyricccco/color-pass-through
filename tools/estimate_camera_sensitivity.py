#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate camera Spectral_Sensitivity (SSF/CSS) from ColorChecker images
under unknown daylight (CIE daylight model), following Jiang et al. WACV 2013.

Changes requested:
1) Reflectance CSV: first row = wavelengths, first col = patch names -> supported.
2) RAW reading: use get_metadata + run_pipeline (rawpy) -> supported.
3) Patch order: no auto detection; user selects via --order {forward,reverse}.
4) Output a PNG that shows 24 rectangles of the ROI used for patch mean computation:
   roi_debug.png (from the first processed image, on the warped/rectified chart).

Outputs:
- estimated_css.npy (float32, [wavelength, R, G, B])
- estimated_css.png
- roi_debug.png   (24 ROI boxes)
- debug_best_cct.json
"""

import argparse
import subprocess
import glob
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import rawpy

# Your modules
from metadata import get_metadata
from isp_pipeline import run_pipeline

# torch optional
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

RAW_EXTS = {".dng", ".nef", ".cr2", ".cr3", ".arw", ".raf", ".rw2", ".srw", ".orf"}
IMG_EXTS = {".tif", ".tiff", ".png", ".npy", ".npz"}
RAW_WB_MODE = "analog_balance"


# -----------------------------
# Reflectance CSV loading
# -----------------------------
def load_reflectance_csv_with_names(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    CSV format:
      first row: 'Color name', then wavelengths (e.g. 380..730)
      first column: patch names (strings)
      remaining cells: reflectance values

    Returns:
      w: (n_w,) wavelengths float64
      R: (n_w, 24) reflectance float64
      names: list[str] length 24
    """
    df = pd.read_csv(path, sep=None, engine="python")  # handles tab/comma

    if df.shape[0] != 24:
        raise ValueError(f"Reflectance CSV must have 24 rows (patches). Got {df.shape[0]} rows.")

    names = df.iloc[:, 0].astype(str).tolist()
    w_cols = df.columns[1:]
    try:
        w = np.array([float(c) for c in w_cols], dtype=np.float64)
    except Exception:
        raise ValueError("Reflectance CSV column headers (except first) must be numeric wavelengths.")

    values = df.iloc[:, 1:].to_numpy(dtype=np.float64)  # (24, n_w)
    R = values.T  # (n_w, 24)
    return w, R, names


# -----------------------------
# Daylight basis + model
# -----------------------------
def load_daylight_basis(daylight_txt: Optional[str], daylight_mat: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIE daylight basis S0,S1,S2 and corresponding wavelengths.
    Supports:
      - daylightScalars.txt: columns [w, S0, S1, S2] OR [*, S0, S1, S2] (w optional)
      - daylight.mat: has daylightScalars (n,4) or S0,S1,S2 (+ optional w)
    """
    if daylight_txt:
        arr = np.loadtxt(daylight_txt)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError("daylightScalars.txt must have at least 4 columns.")
        if arr.shape[1] == 4:
            w = arr[:, 0].astype(np.float64)
            S0, S1, S2 = arr[:, 1].astype(np.float64), arr[:, 2].astype(np.float64), arr[:, 3].astype(np.float64)
        else:
            w = np.arange(arr.shape[0], dtype=np.float64)
            S0, S1, S2 = arr[:, 1].astype(np.float64), arr[:, 2].astype(np.float64), arr[:, 3].astype(np.float64)
        return w, S0, S1, S2

    if daylight_mat:
        d = loadmat(daylight_mat)
        if "daylightScalars" in d:
            arr = np.array(d["daylightScalars"])
            arr = np.squeeze(arr)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                w = arr[:, 0].astype(np.float64)
                S0, S1, S2 = arr[:, 1].astype(np.float64), arr[:, 2].astype(np.float64), arr[:, 3].astype(np.float64)
                return w, S0, S1, S2
        if all(k in d for k in ["S0", "S1", "S2"]):
            S0 = np.squeeze(d["S0"]).astype(np.float64)
            S1 = np.squeeze(d["S1"]).astype(np.float64)
            S2 = np.squeeze(d["S2"]).astype(np.float64)
            w = np.squeeze(d["w"]).astype(np.float64) if "w" in d else np.arange(S0.shape[0], dtype=np.float64)
            return w, S0, S1, S2

    raise ValueError("Provide either --daylight_txt or --daylight_mat")


def cie_daylight_spd(cct: float, w_target: np.ndarray, w_basis: np.ndarray, S0: np.ndarray, S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """
    CIE daylight SPD, matching MATLAB getDaylightScalars(CCT).
    Output interpolated to w_target and normalized at 560nm.
    """
    cct = float(cct)
    if 4000 <= cct <= 7000:
        xD = -4.607e9 / (cct**3) + 2.9678e6 / (cct**2) + 0.09911e3 / cct + 0.244063
    else:
        xD = -2.0064e9 / (cct**3) + 1.9018e6 / (cct**2) + 0.24748e3 / cct + 0.23704

    yD = -3.0 * xD**2 + 2.87 * xD - 0.275
    M1 = (-1.3515 - 1.7703 * xD + 5.9114 * yD) / (0.0241 + 0.2562 * xD - 0.7341 * yD)
    M2 = (0.03 - 31.4424 * xD + 30.0717 * yD) / (0.0241 + 0.2562 * xD - 0.7341 * yD)

    SD = S0 + M1 * S1 + M2 * S2
    l = np.interp(w_target, w_basis, SD).astype(np.float64)

    idx_560 = int(np.argmin(np.abs(w_target - 560.0)))
    if l[idx_560] != 0:
        l = l / l[idx_560]
    return l


# -----------------------------
# Eigenvectors loading
# -----------------------------
def get_eigenvector(refl: np.ndarray, retainE: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    refl: (n_w, n_samples)
    MATLAB: A = refl * refl'
    Return:
      e: (n_w, retainE)  columns are eigenvectors (largest->smallest)
      v: (retainE,)      eigenvalues (largest->smallest)
    """
    A = refl @ refl.T  # (n_w, n_w) symmetric

    # use eigh for symmetric matrices
    vals, vecs = np.linalg.eigh(A)  # ascending
    idx = np.argsort(vals)[::-1]    # descending
    idx = idx[:retainE]

    v = vals[idx]
    e = vecs[:, idx]
    return e, v


def load_eigenvectors(
    *,
    eig_mat: Optional[str] = None,
    eig_npz: Optional[str] = None,
    cam_cmf_dir: Optional[str] = None,
    num_ev: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      E_R, E_G, E_B each shape (num_ev, n_w)
      w_E optional wavelength axis (None if unknown)
    """
    # 1) from folder (closest to MATLAB paper code)
    if cam_cmf_dir is not None:
        files = sorted(glob.glob(os.path.join(cam_cmf_dir, "cmf*.mat")))
        if not files:
            raise FileNotFoundError(f"No cmf*.mat in {cam_cmf_dir}")

        Rs, Gs, Bs = [], [], []
        n_w = None

        for fp in files:
            d = loadmat(fp)
            if not all(k in d for k in ("r", "g", "b")):
                raise ValueError(f"{fp} must contain r,g,b")

            r = np.squeeze(d["r"]).astype(np.float64)
            g = np.squeeze(d["g"]).astype(np.float64)
            b = np.squeeze(d["b"]).astype(np.float64)

            if n_w is None:
                n_w = r.shape[0]
            if r.shape[0] != n_w or g.shape[0] != n_w or b.shape[0] != n_w:
                raise ValueError(f"{fp}: inconsistent curve length")

            # match MATLAB normalization (each curve / max)
            r = r / (np.max(r) + 1e-12)
            g = g / (np.max(g) + 1e-12)
            b = b / (np.max(b) + 1e-12)

            Rs.append(r); Gs.append(g); Bs.append(b)

        redCMF   = np.stack(Rs, axis=1)  # (n_w, n_cam)
        greenCMF = np.stack(Gs, axis=1)
        blueCMF  = np.stack(Bs, axis=1)

        eR, _ = get_eigenvector(redCMF,   retainE=num_ev)  # (n_w, num_ev)
        eG, _ = get_eigenvector(greenCMF, retainE=num_ev)
        eB, _ = get_eigenvector(blueCMF,  retainE=num_ev)

        # our later math uses E_k as (2, n_w)
        E_R = eR.T
        E_G = eG.T
        E_B = eB.T
        return E_R, E_G, E_B, None

    # 2) from npz
    if eig_npz is not None:
        z = np.load(eig_npz, allow_pickle=True)
        E_R = np.array(z["E_R"], dtype=np.float64)
        E_G = np.array(z["E_G"], dtype=np.float64)
        E_B = np.array(z["E_B"], dtype=np.float64)
        w_E = np.array(z["w"], dtype=np.float64) if "w" in z else None
        if E_R.shape[0] != num_ev: E_R = E_R[:num_ev, :]
        if E_G.shape[0] != num_ev: E_G = E_G[:num_ev, :]
        if E_B.shape[0] != num_ev: E_B = E_B[:num_ev, :]
        return E_R, E_G, E_B, w_E

    # 3) from mat (precomputed eRed/eGreen/eBlue)
    if eig_mat is not None:
        d = loadmat(eig_mat)
        eRed = np.array(d["eRed"], dtype=np.float64)   # could be (n_w, k) or (k, n_w)
        eGreen = np.array(d["eGreen"], dtype=np.float64)
        eBlue = np.array(d["eBlue"], dtype=np.float64)

        # force to (n_w, k)
        if eRed.shape[0] == num_ev:   eRed = eRed.T
        if eGreen.shape[0] == num_ev: eGreen = eGreen.T
        if eBlue.shape[0] == num_ev:  eBlue = eBlue.T

        E_R = eRed[:, :num_ev].T
        E_G = eGreen[:, :num_ev].T
        E_B = eBlue[:, :num_ev].T
        w_E = np.squeeze(d["w"]).astype(np.float64) if "w" in d else None
        return E_R, E_G, E_B, w_E

    raise ValueError("Provide one of: cam_cmf_dir / eig_npz / eig_mat")


def interp_basis_to_w(E: np.ndarray, w_E: Optional[np.ndarray], w_target: np.ndarray) -> np.ndarray:
    if w_E is None or np.allclose(w_E, w_target):
        return E
    out = np.zeros((2, w_target.shape[0]), dtype=np.float64)
    for i in range(2):
        out[i, :] = np.interp(w_target, w_E, E[i, :])
    return out


# -----------------------------
# RAW reading using your pipeline
# -----------------------------
def _ensure_rgb(arr, path):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = arr.transpose(1, 2, 0)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{path}: need linear RGB HxWx3 (or 3xHxW)")
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


def _read_linear_rgb_from_raw(path: str) -> np.ndarray:
    rp = rawpy.imread(path)
    meta = get_metadata(rp)

    meta['alpha'] = 0
    meta['ref'] = 'D65'
    meta['demosaic_type'] = 'bilinear'
    meta['color_desc'] = 'RGBG'
    meta['gamma_type'] = 'Rec709'

    raw = rp.raw_image_visible.astype(np.float32)

    if HAS_TORCH:
        if 'wb_matrix' in meta:
            meta['wb_matrix'] = torch.tensor(meta['wb_matrix'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            meta['wb_matrix'] = _maybe_to_cuda(meta['wb_matrix'])
        if 'color_mask' in meta:
            cm = np.asarray(meta['color_mask'])
            meta['color_mask'] = torch.from_numpy(cm).unsqueeze(0).unsqueeze(0)
            meta['color_mask'] = _maybe_to_cuda(meta['color_mask'])
        if 'rgb_xyz_matrix' in meta:
            meta['rgb_xyz_matrix'] = torch.tensor(meta['rgb_xyz_matrix'], dtype=torch.float32).unsqueeze(0)
            meta['rgb_xyz_matrix'] = _maybe_to_cuda(meta['rgb_xyz_matrix'])

        if RAW_WB_MODE == "unity":
            meta["wb_matrix"] = torch.ones((1, 1, 1, 4), dtype=torch.float32)
        elif RAW_WB_MODE == "analog_balance":
            process = subprocess.run(
                ["exiftool", "-j", path],
                check=False,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                raise RuntimeError(f"exiftool failed for {path}: {process.stderr}")
            exifdata = json.loads(process.stdout)[0]
            ab = exifdata.get("AnalogBalance")
            if not ab:
                raise KeyError(f"AnalogBalance missing in {path}")
            vals = [float(x) for x in ab.strip().replace(",", " ").split()]
            if len(vals) < 3:
                raise ValueError(f"Invalid AnalogBalance in {path}: {ab}")
            analog = torch.tensor(vals, dtype=torch.float32)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            variable_wb = torch.stack(
                [analog[0], analog[1], analog[2], analog[1]]
            ).to(device)
            meta["wb_matrix"] = (
                torch.ones(4, dtype=torch.float32, device=device)
                / variable_wb.clamp_min(1e-6)
            ).view(1, 1, 1, 4)
        elif RAW_WB_MODE != "camera":
            raise ValueError(f"Unknown RAW white-balance mode: {RAW_WB_MODE}")
        meta["wb_matrix"] = _maybe_to_cuda(meta["wb_matrix"])

        out = run_pipeline(torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(meta['wb_matrix'].device),
                           meta, 'raw', 'demosaic')
        if torch.is_tensor(out):
            out = out.squeeze(0).detach().cpu().numpy()
        else:
            out = np.asarray(out)
        if out.ndim >= 4:
            out = np.squeeze(out, axis=(0, 1))
    else:
        out = run_pipeline(raw[None, None, ...], meta, 'raw', 'demosaic')

    return _ensure_rgb(out, path)


def read_linear_rgb(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        return _read_linear_rgb_from_raw(path)
    if ext == ".npy":
        return _ensure_rgb(np.load(path), path)
    if ext == ".npz":
        z = np.load(path)
        if len(z.files) == 1:
            return _ensure_rgb(z[z.files[0]], path)
        if "rgb" in z.files:
            return _ensure_rgb(z["rgb"], path)
        raise ValueError(f"{path}: .npz should contain one array or key 'rgb'")
    if ext in {".tif", ".tiff", ".png"}:
        import imageio.v3 as iio
        arr = iio.imread(path)
        return _ensure_rgb(arr, path)
    raise ValueError(f"Unsupported file ext: {ext}")


# -----------------------------
# Geometry + patch ROI extraction
# -----------------------------
def order_corners_xy(corners: np.ndarray) -> np.ndarray:
    pts = corners.astype(np.float64)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def warp_to_canonical(rgb: np.ndarray, corners_xy: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    src = order_corners_xy(corners_xy).astype(np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(rgb, H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return warped


def extract_24_patch_means_and_boxes(
    warped_rgb: np.ndarray,
    grid_cols: int = 6,
    grid_rows: int = 4,
    border_frac: float = 0.08,
    center_frac: float = 0.55,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Returns:
      means: (24,3) in row-major order (top-left -> right, then next row)
      boxes: list of 24 boxes (x0,y0,x1,y1) in warped image coords for the ROI used
    """
    h, w, _ = warped_rgb.shape
    bx = border_frac * w
    by = border_frac * h
    inner_w = w - 2 * bx
    inner_h = h - 2 * by
    cell_w = inner_w / grid_cols
    cell_h = inner_h / grid_rows

    means = []
    boxes = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            x0 = bx + c * cell_w
            y0 = by + r * cell_h

            cx0 = x0 + (1.0 - center_frac) * 0.5 * cell_w
            cy0 = y0 + (1.0 - center_frac) * 0.5 * cell_h
            cx1 = x0 + (1.0 + center_frac) * 0.5 * cell_w
            cy1 = y0 + (1.0 + center_frac) * 0.5 * cell_h

            ix0, iy0 = int(round(cx0)), int(round(cy0))
            ix1, iy1 = int(round(cx1)), int(round(cy1))

            roi = warped_rgb[iy0:iy1, ix0:ix1, :]
            if roi.size == 0:
                raise RuntimeError("Empty ROI. Check corners/warp/border_frac/center_frac.")

            means.append(np.mean(roi.reshape(-1, 3), axis=0))
            boxes.append((ix0, iy0, ix1, iy1))

    return np.stack(means, axis=0), boxes


def save_roi_debug_png(warped_rgb: np.ndarray, boxes: List[Tuple[int, int, int, int]], out_path: str) -> None:
    """
    Save a PNG with 24 ROI rectangles drawn on the warped ColorChecker.
    """
    img = warped_rgb.astype(np.float32)
    # robust normalize for display
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img_disp = np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_disp)
    ax = plt.gca()
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0 + 3, y0 + 15, str(i + 1), color='yellow', fontsize=10, weight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


# -----------------------------
# CSS estimation
# -----------------------------
def solve_channel_c(i_k: np.ndarray, R: np.ndarray, l: np.ndarray, E_k: np.ndarray, delta_lambda: float) -> np.ndarray:
    """
    i_k: (24,)
    R: (n_w,24)
    l: (n_w,)
    E_k: (2,n_w)
    """
    LR = l[:, None] * R           # (n_w,24)
    M = E_k @ LR                  # (2,24)
    target = (i_k / float(delta_lambda)).reshape(-1, 1)   # (24,1)

    s, *_ = np.linalg.lstsq(M.T, target, rcond=None)
    s = s.reshape(-1)             # (2,)
    c_k = (s @ E_k).reshape(-1)   # (n_w,)
    return c_k


def estimate_css_cct_search(
    patch_rgb_24x3: np.ndarray,
    w: np.ndarray, R: np.ndarray,
    E_R: np.ndarray, E_G: np.ndarray, E_B: np.ndarray,
    w_basis: np.ndarray, S0: np.ndarray, S1: np.ndarray, S2: np.ndarray,
    cct_min: int, cct_max: int, cct_step: int,
    delta_lambda: float = 10.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Returns:
      C_best (n_w,3), best_cct, best_err
    """
    I = patch_rgb_24x3.astype(np.float64)

    cct_values = np.arange(cct_min, cct_max + 1, cct_step, dtype=np.float64)
    best_err = np.inf
    best_cct = float(cct_values[0])
    C_best = None

    for cct in cct_values:
        l = cie_daylight_spd(cct, w_target=w, w_basis=w_basis, S0=S0, S1=S1, S2=S2)

        cR = solve_channel_c(I[:, 0], R, l, E_R, delta_lambda)
        cG = solve_channel_c(I[:, 1], R, l, E_G, delta_lambda)
        cB = solve_channel_c(I[:, 2], R, l, E_B, delta_lambda)
        C = np.stack([cR, cG, cB], axis=1)  # (n_w,3)

        LR = l[:, None] * R
        I_hat = (LR.T @ C) * float(delta_lambda)  # (24,3)

        err = float(np.linalg.norm(I - I_hat))
        if err < best_err:
            best_err = err
            best_cct = float(cct)
            C_best = C

    assert C_best is not None
    return C_best, best_cct, best_err


# -----------------------------
# Corners handling
# -----------------------------
def parse_corners_str(s: str) -> np.ndarray:
    parts = s.strip().split(";")
    if len(parts) != 4:
        raise ValueError("Corners must be 4 points: x1,y1;x2,y2;x3,y3;x4,y4")
    pts = []
    for p in parts:
        x, y = p.split(",")
        pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float64)


def load_corners_json(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: np.array(v, dtype=np.float64) for k, v in data.items()}


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets = os.path.join(repository, "assets", "spectral_sensitivity")
    ap.add_argument("--images_dir", required=True, help="Folder containing Colorchecker* files (RAW or npy/png/tif)")
    ap.add_argument("--reflectance_csv", default=os.path.join(assets, "new_color_checker.csv"))
    ap.add_argument("--output_dir", required=True)

    # ap.add_argument("--corners", default="1610,1600;2732,1618;1590,2362;2724,2368", help='Global corners: "x1,y1;x2,y2;x3,y3;x4,y4"')
    ap.add_argument("--corners", default="1187,766;2921,757;1107,1921;2964,1935", help='Global corners: "x1,y1;x2,y2;x3,y3;x4,y4"')
    ap.add_argument("--corners_json", default=None, help="JSON mapping filename->corners; can include 'default'")

    ap.add_argument("--daylight_txt", default=os.path.join(assets, "daylightScalars.txt"))
    ap.add_argument("--daylight_mat", default=None)
    ap.add_argument("--eig_mat", default=None)
    ap.add_argument("--eig_npz", default=None)
    ap.add_argument("--cam_cmf_dir", default=os.path.join(assets, "camSpecSensitivity"))


    ap.add_argument("--warp_w", type=int, default=1200)
    ap.add_argument("--warp_h", type=int, default=800)
    ap.add_argument("--border_frac", type=float, default=0.00)
    ap.add_argument("--center_frac", type=float, default=0.55)

    ap.add_argument("--order", choices=["forward", "reverse"], default="reverse",
                    help="Patch order mapping to reflectance CSV rows. "
                         "forward = top-left->right, row-major; reverse = global reverse (24..1).")

    ap.add_argument("--cct_min", type=int, default=4000)
    ap.add_argument("--cct_max", type=int, default=27000)
    ap.add_argument("--cct_step", type=int, default=100)
    ap.add_argument("--delta_lambda", type=float, default=10.0)
    ap.add_argument(
        "--raw_white_balance",
        choices=["analog_balance", "camera", "unity"],
        default="analog_balance",
        help="Linearization white balance for RAW ColorChecker captures.",
    )

    args = ap.parse_args()
    global RAW_WB_MODE
    RAW_WB_MODE = args.raw_white_balance
    os.makedirs(args.output_dir, exist_ok=True)

    # reflectance (usually 380-730nm)
    w, R, refl_names = load_reflectance_csv_with_names(args.reflectance_csv)

    # daylight basis (400-720nm)
    w_basis, S0, S1, S2 = load_daylight_basis(args.daylight_txt, args.daylight_mat)

    # cropping reflectance and daylight basis to the same range
    w_min = max(w.min(), w_basis.min())
    w_max = min(w.max(), w_basis.max())

    mask = (w >= w_min) & (w <= w_max)
    w = w[mask]
    R = R[mask, :]   # keep (n_w,24)

    # eigenvectors
    E_R, E_G, E_B, w_E = load_eigenvectors(
        eig_mat=args.eig_mat,
        eig_npz=args.eig_npz,
        cam_cmf_dir=args.cam_cmf_dir,
        num_ev=2,
    )
    E_R = interp_basis_to_w(E_R, w_E, w)
    E_G = interp_basis_to_w(E_G, w_E, w)
    E_B = interp_basis_to_w(E_B, w_E, w)

    # corners
    corners_map = load_corners_json(args.corners_json) if args.corners_json else None
    global_corners = parse_corners_str(args.corners) if args.corners else None
    if corners_map is None and global_corners is None:
        raise ValueError("Provide --corners or --corners_json.")

    # collect files
    files = sorted(
        f for f in glob.glob(os.path.join(args.images_dir, "*"))
        if os.path.splitext(f)[1].lower() in RAW_EXTS.union(IMG_EXTS)
        and os.path.basename(f).startswith("Colorchecker")
    )
    if not files:
        raise FileNotFoundError(f"No Colorchecker* files found in {args.images_dir}")

    debug = {
        "created_utc": datetime.now().isoformat() + "Z",
        "order_option": args.order,
        "reflectance_names_csv": refl_names,
        "images": {}
    }

    per_img_css = []
    saved_roi_debug = False

    for fp in files:
        fname = os.path.basename(fp)

        # select corners: per-image > default > global
        if corners_map is not None:
            if fname in corners_map:
                corners = corners_map[fname]
            elif "default" in corners_map:
                corners = corners_map["default"]
            elif global_corners is not None:
                corners = global_corners
            else:
                raise ValueError(f"No corners for {fname}.")
        else:
            corners = global_corners

        rgb = read_linear_rgb(fp)
        warped = warp_to_canonical(rgb, corners, out_w=args.warp_w, out_h=args.warp_h)

        patch_means, boxes = extract_24_patch_means_and_boxes(
            warped,
            border_frac=args.border_frac,
            center_frac=args.center_frac
        )  # forward row-major extraction

        # Save ONE ROI debug PNG from the first processed image
        if not saved_roi_debug:
            out_roi = os.path.join(args.output_dir, "roi_debug.png")
            save_roi_debug_png(warped, boxes, out_roi)
            saved_roi_debug = True

        # Apply user-specified mapping order
        if args.order == "reverse":
            patch_means = patch_means[::-1, :]

        # Estimate CSS for this image
        C_best, best_cct, best_err = estimate_css_cct_search(
            patch_means, w, R,
            E_R, E_G, E_B,
            w_basis, S0, S1, S2,
            args.cct_min, args.cct_max, args.cct_step,
            delta_lambda=args.delta_lambda
        )

        # normalize & clip (scale ambiguity)
        C_best = C_best / (np.max(C_best) + 1e-12)
        C_best = np.clip(C_best, 0.0, None)

        per_img_css.append(C_best)
        debug["images"][fname] = {
            "best_cct": float(best_cct),
            "best_err": float(best_err)
        }

    # average across images
    C_avg = np.mean(np.stack(per_img_css, axis=0), axis=0)
    C_avg = C_avg / (np.max(C_avg) + 1e-12)
    C_avg = np.clip(C_avg, 0.0, None).astype(np.float32)

    w_f32 = w.astype(np.float32).reshape(-1, 1)          # (n_w,1)
    css_wrgb = np.concatenate([w_f32, C_avg], axis=1)    # (n_w,4)

    out_npy = os.path.join(args.output_dir, "estimated_css.npy")
    np.save(out_npy, css_wrgb)

    out_png = os.path.join(args.output_dir, "estimated_css.png")
    plt.figure(figsize=(10, 4))
    plt.plot(w, C_avg[:, 0], label="R", color="r")
    plt.plot(w, C_avg[:, 1], label="G", color="g")
    plt.plot(w, C_avg[:, 2], label="B", color="b")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative sensitivity (normalized)")
    plt.title("Estimated Camera Spectral Sensitivity Functions (unknown daylight)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    out_json = os.path.join(args.output_dir, "debug_best_cct.json")
    debug["summary"] = {
        "num_images": len(files),
        "mean_best_cct": float(np.mean([v["best_cct"] for v in debug["images"].values()])),
        "output_npy": out_npy,
        "output_png": out_png,
        "output_roi_debug_png": os.path.join(args.output_dir, "roi_debug.png")
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2, ensure_ascii=False)

    print("[OK] Saved:", out_npy)
    print("[OK] Saved:", out_png)
    print("[OK] Saved ROI debug PNG:", os.path.join(args.output_dir, "roi_debug.png"))
    print("[OK] Saved:", out_json)


if __name__ == "__main__":
    main()
