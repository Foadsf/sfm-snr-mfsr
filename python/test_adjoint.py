#!/usr/bin/env python3
"""
Unit Test: Exact Adjoint Verification for Projective Warping + PSF + Decimation Operator.
"""

import numpy as np
import cv2

def get_intrinsics(h, w, fov_deg=50.0):
    f = 0.5 * w / np.tan(0.5 * np.deg2rad(fov_deg))
    return np.array([
        [f, 0, (w - 1) / 2.0],
        [0, f, (h - 1) / 2.0],
        [0, 0, 1.0]
    ], dtype=np.float64)

def project_ref_to_view(u_hr, v_hr, depth_hr, K_hr, R_k, t_k, scale=2):
    fx, fy = K_hr[0, 0], K_hr[1, 1]
    cx, cy = K_hr[0, 2], K_hr[1, 2]
    
    X0 = (u_hr - cx) / fx * depth_hr
    Y0 = (v_hr - cy) / fy * depth_hr
    Z0 = depth_hr
    
    Xk = R_k[0, 0] * X0 + R_k[0, 1] * Y0 + R_k[0, 2] * Z0 + t_k[0]
    Yk = R_k[1, 0] * X0 + R_k[1, 1] * Y0 + R_k[1, 2] * Z0 + t_k[1]
    Zk = R_k[2, 0] * X0 + R_k[2, 1] * Y0 + R_k[2, 2] * Z0 + t_k[2]
    
    u_k_hr = fx * (Xk / Zk) + cx
    v_k_hr = fy * (Yk / Zk) + cy
    
    u_k_lr = (u_k_hr + 0.5) / scale - 0.5
    v_k_lr = (v_k_hr + 0.5) / scale - 0.5
    return u_k_lr.astype(np.float32), v_k_lr.astype(np.float32)

def precompute_splat_weights(u_k, v_k, lr_h, lr_w):
    u0 = np.floor(u_k).astype(np.int32)
    v0 = np.floor(v_k).astype(np.int32)
    du = u_k - u0
    dv = v_k - v0
    
    valid = (u0 >= 0) & (u0 < lr_w - 1) & (v0 >= 0) & (v0 < lr_h - 1)
    w00 = ((1.0 - du) * (1.0 - dv) * valid).ravel().astype(np.float64)
    w10 = (du * (1.0 - dv) * valid).ravel().astype(np.float64)
    w01 = ((1.0 - du) * dv * valid).ravel().astype(np.float64)
    w11 = (du * dv * valid).ravel().astype(np.float64)
    
    u0_c = np.clip(u0, 0, lr_w - 2)
    v0_c = np.clip(v0, 0, lr_h - 2)
    
    idx00 = (v0_c * lr_w + u0_c).ravel()
    idx10 = (v0_c * lr_w + (u0_c + 1)).ravel()
    idx01 = ((v0_c + 1) * lr_w + u0_c).ravel()
    idx11 = ((v0_c + 1) * lr_w + (u0_c + 1)).ravel()
    
    n_lr = lr_h * lr_w
    norm = (
        np.bincount(idx00, weights=w00, minlength=n_lr) +
        np.bincount(idx10, weights=w10, minlength=n_lr) +
        np.bincount(idx01, weights=w01, minlength=n_lr) +
        np.bincount(idx11, weights=w11, minlength=n_lr)
    ).reshape((lr_h, lr_w))
    
    return idx00, idx10, idx01, idx11, w00, w10, w01, w11, norm, u_k, v_k, valid

def forward_A(X_hr, splat_data, psf_sigma=0.45):
    idx00, idx10, idx01, idx11, w00, w10, w01, w11, norm, _, _, _ = splat_data
    lr_h, lr_w = norm.shape
    n_lr = lr_h * lr_w
    
    # 1. Optical PSF blur H
    ksize = int(6 * psf_sigma + 1) | 1
    blurred = cv2.GaussianBlur(X_hr.astype(np.float64), (ksize, ksize), sigmaX=psf_sigma, sigmaY=psf_sigma, borderType=cv2.BORDER_REFLECT)
    b_flat = blurred.reshape((-1, 3))
    
    # 2. Geometric Warp W + Decimation D
    out_lr = np.empty((lr_h, lr_w, 3), dtype=np.float64)
    for c in range(3):
        bc = b_flat[:, c]
        acc = (
            np.bincount(idx00, weights=bc * w00, minlength=n_lr) +
            np.bincount(idx10, weights=bc * w10, minlength=n_lr) +
            np.bincount(idx01, weights=bc * w01, minlength=n_lr) +
            np.bincount(idx11, weights=bc * w11, minlength=n_lr)
        ).reshape((lr_h, lr_w))
        out_lr[:, :, c] = acc / (norm + 1e-6)
    return out_lr

def adjoint_AT(Y_lr, splat_data, psf_sigma=0.45):
    idx00, idx10, idx01, idx11, w00, w10, w01, w11, norm, u_k, _, _ = splat_data
    hr_h, hr_w = u_k.shape
    
    # 1. Exact Adjoint Decimation + Splatting D^T W^T
    norm_res = (Y_lr / (norm[:, :, None] + 1e-6)).reshape((-1, 3))
    pulled = np.zeros((hr_h * hr_w, 3), dtype=np.float64)
    for c in range(3):
        yc = norm_res[:, c]
        pulled[:, c] = (
            w00 * yc[idx00] +
            w10 * yc[idx10] +
            w01 * yc[idx01] +
            w11 * yc[idx11]
        )
    pulled = pulled.reshape((hr_h, hr_w, 3))
    
    # 2. Adjoint PSF blur H^T (Symmetric Gaussian kernel -> H^T = H)
    ksize = int(6 * psf_sigma + 1) | 1
    out_hr = cv2.GaussianBlur(pulled, (ksize, ksize), sigmaX=psf_sigma, sigmaY=psf_sigma, borderType=cv2.BORDER_REFLECT)
    return out_hr

def test_adjoint():
    hr_h, hr_w = 200, 200
    scale = 2
    lr_h, lr_w = hr_h // scale, hr_w // scale
    
    K_hr = get_intrinsics(hr_h, hr_w)
    depth_gt = np.full((hr_h, hr_w), 4.5, dtype=np.float32)
    
    R = np.eye(3, dtype=np.float64)
    t = np.array([0.15, -0.10, 0.05], dtype=np.float64)
    v_hr, u_hr = np.mgrid[0:hr_h, 0:hr_w].astype(np.float32)
    uk, vk = project_ref_to_view(u_hr, v_hr, depth_gt, K_hr, R, t, scale=scale)
    splat_data = precompute_splat_weights(uk, vk, lr_h, lr_w)
    
    rng = np.random.default_rng(42)
    X_rnd = rng.standard_normal((hr_h, hr_w, 3))
    Y_rnd = rng.standard_normal((lr_h, lr_w, 3))
    
    AX = forward_A(X_rnd, splat_data)
    ATY = adjoint_AT(Y_rnd, splat_data)
    
    lhs = np.sum(AX * Y_rnd)
    rhs = np.sum(X_rnd * ATY)
    
    rel_diff = np.abs(lhs - rhs) / max(np.abs(lhs), np.abs(rhs))
    print(f"Adjoint Dot-Product Test:")
    print(f"  <A X, Y>     = {lhs:.10f}")
    print(f"  <X, A^T Y>   = {rhs:.10f}")
    print(f"  Relative Diff = {rel_diff:.4e}")
    
    assert rel_diff < 1e-6, f"Adjoint dot-product test failed! rel_diff={rel_diff}"
    print("  [PASS] Adjoint operator matches forward operator to machine precision (< 1e-6)!")

if __name__ == "__main__":
    test_adjoint()
