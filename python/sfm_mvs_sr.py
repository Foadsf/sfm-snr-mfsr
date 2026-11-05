#!/usr/bin/env python3
"""
Standalone Proof-of-Concept: Structure from Motion (SfM) + Multi-View Stereo (MVS)
for Multi-Frame Super-Resolution (MFSR) and SNR Enhancement.
"""

import os
import time
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.interpolate import RBFInterpolator
from scipy.sparse import lil_matrix

cv2.ocl.setUseOpenCL(True)


def generate_synthetic_3d_scene(hr_h=400, hr_w=400):
    y, x = np.mgrid[0:hr_h, 0:hr_w].astype(np.float32)
    xn, yn = (x - hr_w / 2) / (hr_w / 2), (y - hr_h / 2) / (hr_h / 2)
    img = np.zeros((hr_h, hr_w, 3), dtype=np.float32)
    r2 = xn ** 2 + yn ** 2
    freq_pattern = 0.5 + 0.5 * np.cos(40.0 * r2 + 15.0 * xn * yn)
    checker = ((np.floor(x / 16) + np.floor(y / 16)) % 2) * 0.3
    fine_grid = ((np.floor(x / 4) % 2 == 0) | (np.floor(y / 4) % 2 == 0)).astype(np.float32) * 0.15

    img[:, :, 0] = np.clip(0.2 + 0.5 * freq_pattern + checker, 0, 1)
    img[:, :, 1] = np.clip(0.3 + 0.4 * np.sin(25 * xn) * np.cos(25 * yn) + fine_grid, 0, 1)
    img[:, :, 2] = np.clip(0.2 + 0.6 * (1 - r2 / 2) + 0.2 * np.cos(50 * yn), 0, 1)

    img_u8 = (img * 255).astype(np.uint8)
    cv2.putText(img_u8, "SfM-MVS SUPER-RES", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_u8, "1234567890 ABCDEF", (40, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 220, 50), 2, cv2.LINE_AA)
    cv2.circle(img_u8, (200, 200), 70, (40, 240, 120), 2, cv2.LINE_AA)
    cv2.rectangle(img_u8, (90, 130), (310, 270), (240, 60, 60), 2, cv2.LINE_AA)

    rng = np.random.default_rng(42)
    for _ in range(120):
        cx, cy = int(rng.integers(20, hr_w - 20)), int(rng.integers(20, hr_h - 20))
        rad = int(rng.integers(4, 14))
        color = tuple(int(c) for c in rng.integers(30, 250, size=3))
        cv2.circle(img_u8, (cx, cy), rad, color, -1, cv2.LINE_AA)
        cv2.circle(img_u8, (cx, cy), rad, (0, 0, 0), 1, cv2.LINE_AA)

    hr_clean = img_u8.astype(np.float32) / 255.0
    depth_gt = (
        4.2
        - 0.9 * np.exp(-2.5 * ((xn + 0.25) ** 2 + (yn + 0.2) ** 2))
        + 0.8 * np.exp(-2.5 * ((xn - 0.3) ** 2 + (yn - 0.25) ** 2))
        + 0.35 * np.sin(2.5 * xn) * np.cos(2.5 * yn)
    ).astype(np.float32)
    return hr_clean, depth_gt


def get_intrinsics(h, w, fov_deg=50.0):
    f = 0.5 * w / np.tan(0.5 * np.deg2rad(fov_deg))
    return np.array([[f, 0, (w - 1) / 2.0], [0, f, (h - 1) / 2.0], [0, 0, 1.0]], dtype=np.float64)


def project_ref_to_view(u_hr, v_hr, depth_hr, K_hr, R_k, t_k, scale=2):
    fx, fy, cx, cy = K_hr[0, 0], K_hr[1, 1], K_hr[0, 2], K_hr[1, 2]
    X0 = (u_hr - cx) / fx * depth_hr
    Y0 = (v_hr - cy) / fy * depth_hr
    Z0 = depth_hr
    Xk = R_k[0, 0] * X0 + R_k[0, 1] * Y0 + R_k[0, 2] * Z0 + t_k[0]
    Yk = R_k[1, 0] * X0 + R_k[1, 1] * Y0 + R_k[1, 2] * Z0 + t_k[1]
    Zk = np.maximum(1e-5, R_k[2, 0] * X0 + R_k[2, 1] * Y0 + R_k[2, 2] * Z0 + t_k[2])
    u_k_hr = fx * (Xk / Zk) + cx
    v_k_hr = fy * (Yk / Zk) + cy
    return ((u_k_hr + 0.5) / scale - 0.5).astype(np.float32), ((v_k_hr + 0.5) / scale - 0.5).astype(np.float32)


def synthesize_multiview_lr_dataset(hr_clean, depth_gt, num_views=12, scale=2, noise_sigma=10.0 / 255.0):
    hr_h, hr_w, _ = hr_clean.shape
    lr_h, lr_w = hr_h // scale, hr_w // scale
    K_hr = get_intrinsics(hr_h, hr_w)
    K_lr = K_hr.copy()
    K_lr[0, 0] /= scale
    K_lr[1, 1] /= scale
    K_lr[0, 2] = (K_hr[0, 2] + 0.5) / scale - 0.5
    K_lr[1, 2] = (K_hr[1, 2] + 0.5) / scale - 0.5

    poses_gt, lr_noisy_views, lr_clean_views = [], [], []
    v_grid, u_grid = np.mgrid[0:hr_h, 0:hr_w].astype(np.float32)
    rng = np.random.default_rng(1234)

    for k in range(num_views):
        if k == 0:
            R_k, t_k = np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        else:
            angle = 2.0 * np.pi * (k - 1) / (num_views - 1)
            radius = 0.22 + 0.05 * ((k % 2) - 0.5)
            tx, ty, tz = radius * np.cos(angle), radius * np.sin(angle), 0.06 * np.sin(2 * angle)
            t_k = np.array([tx, ty, tz], dtype=np.float64)
            rot_vec = np.array([-ty / 5.0, tx / 5.0, 0.015 * np.cos(angle)], dtype=np.float64)
            R_k, _ = cv2.Rodrigues(rot_vec)
        poses_gt.append((R_k, t_k))

        u_ref, v_ref = u_grid.copy(), v_grid.copy()
        for _ in range(5):
            d_est = cv2.remap(depth_gt, u_ref, v_ref, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            u_k_hr, v_k_hr = project_ref_to_view(u_ref, v_ref, d_est, K_hr, R_k, t_k, scale=1)
            u_ref += (u_grid - u_k_hr)
            v_ref += (v_grid - v_k_hr)

        warped_hr = cv2.remap(hr_clean, u_ref, v_ref, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        blurred_hr = cv2.GaussianBlur(warped_hr, (0, 0), sigmaX=0.85, sigmaY=0.85)
        lr_clean = cv2.resize(blurred_hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
        lr_clean_views.append(lr_clean)
        noise = rng.normal(0.0, noise_sigma, size=lr_clean.shape).astype(np.float32)
        lr_noisy_views.append(np.clip(lr_clean + noise, 0.0, 1.0))

    return lr_noisy_views, lr_clean_views, poses_gt, K_lr, K_hr


def run_sfm_and_ba(lr_views, K_lr, preprocess_mode='sr_denoise'):
    num_views = len(lr_views)
    sift = cv2.SIFT_create(nfeatures=8000, nOctaveLayers=4, contrastThreshold=0.008, edgeThreshold=18, sigma=1.2)
    keypoints_list, descriptors_list = [], []

    for img in lr_views:
        gray_u8 = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
        if preprocess_mode == 'raw':
            proc = cv2.cvtColor(gray_u8, cv2.COLOR_BGR2GRAY)
            scale_factor = 1.0
        else:
            denoised = cv2.GaussianBlur(gray_u8, (3, 3), 0.7)
            up = cv2.resize(denoised, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            proc = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            scale_factor = 2.0

        kps, desc = sift.detectAndCompute(proc, None)
        scaled_kps = [
            cv2.KeyPoint(
                x=(kp.pt[0] + 0.5) / scale_factor - 0.5,
                y=(kp.pt[1] + 0.5) / scale_factor - 0.5,
                size=kp.size / scale_factor, angle=kp.angle,
                response=kp.response, octave=kp.octave, class_id=kp.class_id
            ) for kp in kps
        ]
        if desc is not None:
            desc /= (np.linalg.norm(desc, ord=1, axis=1, keepdims=True) + 1e-7)
            desc = np.sqrt(desc)
        keypoints_list.append(scaled_kps)
        descriptors_list.append(desc)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches_to_ref = {
        k: [m for m, n in bf.knnMatch(descriptors_list[0], descriptors_list[k], k=2) if m.distance < 0.75 * n.distance]
        for k in range(1, num_views)
    }

    m01 = matches_to_ref[1]
    pts0 = np.float64([keypoints_list[0][m.queryIdx].pt for m in m01])
    pts1 = np.float64([keypoints_list[1][m.trainIdx].pt for m in m01])
    E, inlier_mask = cv2.findEssentialMat(pts0, pts1, K_lr, method=cv2.USAC_MAGSAC, prob=0.9999, threshold=0.4)
    _, R1, t1, inlier_mask = cv2.recoverPose(E, pts0, pts1, K_lr, mask=inlier_mask)

    P0 = K_lr @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K_lr @ np.hstack([R1, t1])
    inliers01 = inlier_mask.ravel() > 0
    pts4d = cv2.triangulatePoints(P0, P1, pts0[inliers01].T, pts1[inliers01].T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    m01_in = [m for m, inl in zip(m01, inliers01) if inl]

    kp0_to_3d, valid_3d, observations = {}, [], []
    for idx_in, m in enumerate(m01_in):
        if 1.0 < pts3d[idx_in, 2] < 100.0:
            pt_id = len(valid_3d)
            valid_3d.append(pts3d[idx_in])
            kp0_to_3d[m.queryIdx] = pt_id
            u0, v0 = keypoints_list[0][m.queryIdx].pt
            u1, v1 = keypoints_list[1][m.trainIdx].pt
            observations.append((0, pt_id, u0, v0))
            observations.append((1, pt_id, u1, v1))

    valid_3d = np.array(valid_3d, dtype=np.float64)
    poses_est = [(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)), (R1, t1.ravel())]

    for k in range(2, num_views):
        obj_pts, img_pts, matched_pt_ids = [], [], []
        for m in matches_to_ref[k]:
            if m.queryIdx in kp0_to_3d:
                pt_id = kp0_to_3d[m.queryIdx]
                obj_pts.append(valid_3d[pt_id])
                img_pts.append(keypoints_list[k][m.trainIdx].pt)
                matched_pt_ids.append(pt_id)
        obj_pts, img_pts = np.array(obj_pts, dtype=np.float64), np.array(img_pts, dtype=np.float64)
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K_lr, None, iterationsCount=5000, reprojectionError=0.6, confidence=0.9999, flags=cv2.SOLVEPNP_SQPNP
        )
        inl_idx = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts[inl_idx], img_pts[inl_idx], K_lr, None, rvec, tvec)
        for idx in inl_idx:
            observations.append((k, matched_pt_ids[idx], img_pts[idx][0], img_pts[idx][1]))
        R_k, _ = cv2.Rodrigues(rvec)
        poses_est.append((R_k, tvec.ravel()))

    obs_cam = np.array([o[0] for o in observations], dtype=np.int32)
    obs_pt  = np.array([o[1] for o in observations], dtype=np.int32)
    obs_uv  = np.array([[o[2], o[3]] for o in observations], dtype=np.float64)
    n_cams, n_pts = num_views - 1, len(valid_3d)
    x0 = np.zeros(n_cams * 6 + n_pts * 3, dtype=np.float64)
    for k in range(1, num_views):
        rvec, _ = cv2.Rodrigues(poses_est[k][0])
        x0[(k - 1) * 6:(k - 1) * 6 + 3] = rvec.ravel()
        x0[(k - 1) * 6 + 3:(k - 1) * 6 + 6] = poses_est[k][1]
    x0[n_cams * 6:] = valid_3d.ravel()

    fx, fy, cx, cy = K_lr[0, 0], K_lr[1, 1], K_lr[0, 2], K_lr[1, 2]

    def ba_residuals(params):
        cam_params = params[:n_cams * 6].reshape((n_cams, 6))
        pts_3d = params[n_cams * 6:].reshape((n_pts, 3))
        Rs, ts = [np.eye(3)], [np.zeros(3)]
        for c in range(n_cams):
            R_c, _ = cv2.Rodrigues(cam_params[c, :3])
            Rs.append(R_c)
            ts.append(cam_params[c, 3:6])
        Rs, ts = np.array(Rs), np.array(ts)
        P_cam = np.einsum('nij,nj->ni', Rs[obs_cam], pts_3d[obs_pt]) + ts[obs_cam]
        u_proj = fx * (P_cam[:, 0] / P_cam[:, 2]) + cx
        v_proj = fy * (P_cam[:, 1] / P_cam[:, 2]) + cy
        return np.column_stack([u_proj - obs_uv[:, 0], v_proj - obs_uv[:, 1]]).ravel()

    A_sparse = lil_matrix((len(obs_cam) * 2, len(x0)), dtype=int)
    for i, (c_idx, p_idx) in enumerate(zip(obs_cam, obs_pt)):
        if c_idx > 0:
            A_sparse[2 * i:2 * i + 2, (c_idx - 1) * 6:(c_idx - 1) * 6 + 6] = 1
        A_sparse[2 * i:2 * i + 2, n_cams * 6 + p_idx * 3:n_cams * 6 + p_idx * 3 + 3] = 1

    opt = least_squares(ba_residuals, x0, jac_sparsity=A_sparse, loss='huber', f_scale=0.2, max_nfev=100, verbose=0)
    cam_opt = opt.x[:n_cams * 6].reshape((n_cams, 6))
    pts_opt = opt.x[n_cams * 6:].reshape((n_pts, 3))
    poses_ba = [(np.eye(3), np.zeros(3))]
    for c in range(n_cams):
        R_c, _ = cv2.Rodrigues(cam_opt[c, :3])
        poses_ba.append((R_c, cam_opt[c, 3:6]))
    return poses_ba, pts_opt, float(np.sqrt(np.mean(opt.fun ** 2)))


def estimate_dense_mvs_depth(lr_views, poses_est, pts_3d, K_hr, scale=2):
    hr_h, hr_w = lr_views[0].shape[0] * scale, lr_views[0].shape[1] * scale
    fx_hr, fy_hr, cx_hr, cy_hr = K_hr[0, 0], K_hr[1, 1], K_hr[0, 2], K_hr[1, 2]
    u_pts = fx_hr * (pts_3d[:, 0] / pts_3d[:, 2]) + cx_hr
    v_pts = fy_hr * (pts_3d[:, 1] / pts_3d[:, 2]) + cy_hr
    z_pts = pts_3d[:, 2]
    z_med = np.median(z_pts)
    z_mad = np.median(np.abs(z_pts - z_med))
    valid = (u_pts >= 0) & (u_pts < hr_w) & (v_pts >= 0) & (v_pts < hr_h) & (np.abs(z_pts - z_med) < 4.5 * z_mad)

    grid_y, grid_x = np.mgrid[0:hr_h:4, 0:hr_w:4].astype(np.float32)
    rbf = RBFInterpolator(
        np.column_stack([u_pts[valid], v_pts[valid]]), z_pts[valid],
        kernel='thin_plate_spline', smoothing=np.sum(valid) * 0.02
    )
    d_coarse = rbf(np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape).astype(np.float32)
    d_prior = cv2.resize(d_coarse, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

    lr_smooth = [cv2.GaussianBlur(img, (3, 3), 0.6) for img in lr_views]
    ref_up = cv2.resize(lr_smooth[0], (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
    num_planes = 11
    offsets = np.linspace(-0.06 * z_med, 0.06 * z_med, num_planes, dtype=np.float32)
    v_hr, u_hr = np.mgrid[0:hr_h, 0:hr_w].astype(np.float32)
    cost_volume = np.zeros((num_planes, hr_h, hr_w), dtype=np.float32)

    for p_idx, dz in enumerate(offsets):
        d_hyp = d_prior + dz
        err_acc = np.zeros((hr_h, hr_w), dtype=np.float32)
        for k in range(1, len(lr_views)):
            R_k, t_k = poses_est[k]
            uk, vk = project_ref_to_view(u_hr, v_hr, d_hyp, K_hr, R_k, t_k, scale=scale)
            warped_k = cv2.remap(lr_smooth[k], uk, vk, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            err_acc += np.minimum(np.mean(np.abs(warped_k - ref_up), axis=2), 0.15)
        cost_volume[p_idx] = cv2.boxFilter(err_acc / (len(lr_views) - 1), -1, (9, 9))

    idx_c = np.clip(np.argmin(cost_volume, axis=0), 1, num_planes - 2)
    rows, cols = np.ogrid[:hr_h, :hr_w]
    c0, c1, c2 = cost_volume[idx_c - 1, rows, cols], cost_volume[idx_c, rows, cols], cost_volume[idx_c + 1, rows, cols]
    denom = 2.0 * (c0 - 2.0 * c1 + c2)
    sub_offset = np.clip(np.where(np.abs(denom) > 1e-6, (c0 - c2) / denom, 0.0), -1.0, 1.0)
    d_refined = d_prior + offsets[idx_c] + sub_offset * (offsets[1] - offsets[0])
    return cv2.GaussianBlur(d_refined.astype(np.float32), (11, 11), 2.5)


def precompute_splat_weights_fast(u_k, v_k, lr_h, lr_w):
    u0, v0 = np.floor(u_k).astype(np.int32), np.floor(v_k).astype(np.int32)
    du, dv = u_k - u0, v_k - v0
    valid = (u0 >= 0) & (u0 < lr_w - 1) & (v0 >= 0) & (v0 < lr_h - 1)
    w00 = ((1.0 - du) * (1.0 - dv) * valid).ravel()
    w10 = (du * (1.0 - dv) * valid).ravel()
    w01 = ((1.0 - du) * dv * valid).ravel()
    w11 = (du * dv * valid).ravel()
    u0_c, v0_c = np.clip(u0, 0, lr_w - 2), np.clip(v0, 0, lr_h - 2)
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
    ).astype(np.float32).reshape((lr_h, lr_w))
    return (idx00, idx10, idx01, idx11, w00, w10, w01, w11, norm, u_k, v_k, valid)


def forward_project_view_fast(hr_img, splat_data, psf_sigma=0.45):
    idx00, idx10, idx01, idx11, w00, w10, w01, w11, norm, _, _, _ = splat_data
    lr_h, lr_w = norm.shape
    n_lr = lr_h * lr_w
    blurred = cv2.GaussianBlur(hr_img, (0, 0), sigmaX=psf_sigma, sigmaY=psf_sigma)
    b_flat = blurred.reshape((-1, 3))
    out_lr = np.empty((lr_h, lr_w, 3), dtype=np.float32)
    for c in range(3):
        bc = b_flat[:, c]
        acc = (
            np.bincount(idx00, weights=bc * w00, minlength=n_lr) +
            np.bincount(idx10, weights=bc * w10, minlength=n_lr) +
            np.bincount(idx01, weights=bc * w01, minlength=n_lr) +
            np.bincount(idx11, weights=bc * w11, minlength=n_lr)
        )
        out_lr[:, :, c] = acc.reshape((lr_h, lr_w)) / (norm + 1e-6)
    return out_lr, norm > 1e-3


def adjoint_backproject_view(lr_residual, splat_data, psf_sigma=0.45):
    _, _, _, _, _, _, _, _, norm, u_k, v_k, valid = splat_data
    norm_res = lr_residual / (norm[:, :, None] + 1e-6)
    pulled = cv2.remap(norm_res, u_k, v_k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pulled *= valid[:, :, None]
    return cv2.GaussianBlur(pulled, (0, 0), sigmaX=psf_sigma, sigmaY=psf_sigma)


def bilateral_tv_grad(X, alpha=0.6):
    grad = np.zeros_like(X)
    eps = 1e-2
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            weight = alpha ** (abs(dx) + abs(dy))
            shifted = np.roll(np.roll(X, dy, axis=0), dx, axis=1)
            diff = X - shifted
            charb_deriv = diff / np.sqrt(diff * diff + eps * eps)
            back_shifted = np.roll(np.roll(charb_deriv, -dy, axis=0), -dx, axis=1)
            grad += weight * (charb_deriv - back_shifted)
    return grad


def reconstruct_mfsr_exact_psf(lr_views, poses_est, depth_est, K_hr, scale=2, num_iters=25, lam_btv=0.001):
    hr_h, hr_w = lr_views[0].shape[0] * scale, lr_views[0].shape[1] * scale
    lr_h, lr_w, _ = lr_views[0].shape
    v_hr, u_hr = np.mgrid[0:hr_h, 0:hr_w].astype(np.float32)
    splats, warped_stack, valid_masks = [], [], []

    for k, (R_k, t_k) in enumerate(poses_est):
        uk, vk = project_ref_to_view(u_hr, v_hr, depth_est, K_hr, R_k, t_k, scale=scale)
        sp = precompute_splat_weights_fast(uk, vk, lr_h, lr_w)
        splats.append(sp)
        warped_stack.append(cv2.remap(lr_views[k], uk, vk, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE))
        valid_masks.append(sp[11].astype(np.float32))

    warped_stack = np.array(warped_stack)
    med_img = np.median(warped_stack, axis=0)
    abs_dev = np.abs(warped_stack - med_img[None, ...])
    mad = np.median(abs_dev, axis=0) + 1e-4
    conf_weights = np.exp(-0.5 * (abs_dev / (2.2 * mad[None, ...])) ** 2) * np.array(valid_masks)[..., None]
    conf_weights[0] = 1.5
    X_init = np.sum(warped_stack * conf_weights, axis=0) / (np.sum(conf_weights, axis=0) + 1e-6)

    X, Y = X_init.copy(), X_init.copy()
    t_nesterov, step_size = 1.0, 1.1

    for _ in range(num_iters):
        grad_data = np.zeros_like(Y)
        weight_sum = np.zeros((hr_h, hr_w, 3), dtype=np.float32)
        for k in range(len(lr_views)):
            pred_lr, mask_lr = forward_project_view_fast(Y, splats[k], psf_sigma=0.45)
            diff_lr = (pred_lr - lr_views[k]) * mask_lr[:, :, None]
            grad_k = adjoint_backproject_view(diff_lr, splats[k], psf_sigma=0.45)
            grad_data += conf_weights[k] * grad_k
            weight_sum += conf_weights[k]
        grad_data /= (weight_sum + 1e-6)
        grad_reg = bilateral_tv_grad(Y)
        X_next = np.clip(Y - step_size * (grad_data + lam_btv * grad_reg), 0.0, 1.0)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_nesterov * t_nesterov))
        Y = np.clip(X_next + ((t_nesterov - 1.0) / t_next) * (X_next - X), 0.0, 1.0)
        X, t_nesterov = X_next, t_next

    return X, X_init


def compute_metrics(gt, est):
    mse = float(np.mean((gt - est) ** 2))
    psnr = 10.0 * np.log10(1.0 / (mse + 1e-12))
    snr = 10.0 * np.log10(float(np.sum(gt ** 2)) / (float(np.sum((gt - est) ** 2)) + 1e-12))
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_vals = []
    for c in range(3):
        x, y = gt[:, :, c], est[:, :, c]
        mu_x, mu_y = cv2.GaussianBlur(x, (11, 11), 1.5), cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
        sigma_x2 = cv2.GaussianBlur(x ** 2, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y ** 2, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        ssim_vals.append(np.mean(((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))))
    return psnr, snr, float(np.mean(ssim_vals))


if __name__ == "__main__":
    t0 = time.time()
    hr_clean, depth_gt = generate_synthetic_3d_scene()
    lr_noisy, lr_clean, poses_gt, K_lr, K_hr = synthesize_multiview_lr_dataset(hr_clean, depth_gt)

    poses_raw, pts_raw, rmse_raw = run_sfm_and_ba(lr_noisy, K_lr, 'raw')
    poses_sr, pts_sr, rmse_sr = run_sfm_and_ba(lr_noisy, K_lr, 'sr_denoise')
    depth_mvs = estimate_dense_mvs_depth(lr_noisy, poses_sr, pts_sr, K_hr, scale=2)
    hr_sr_map, hr_sr_init = reconstruct_mfsr_exact_psf(lr_noisy, poses_sr, depth_mvs, K_hr, scale=2)

    m = 16
    gt_c = hr_clean[m:-m, m:-m]
    bic_noisy = cv2.resize(lr_noisy[0], (400, 400), interpolation=cv2.INTER_CUBIC)[m:-m, m:-m]
    bic_clean = cv2.resize(lr_clean[0], (400, 400), interpolation=cv2.INTER_CUBIC)[m:-m, m:-m]

    print(f"SfM Landmarks: Raw={len(pts_raw)} -> SR+Denoised={len(pts_sr)} (+{(len(pts_sr)/len(pts_raw)-1)*100:.1f}%)")
    for name, img in [
        ("1. Single-View Noisy Bicubic       ", bic_noisy),
        ("2. Oracle Clean Single-View Bicubic", bic_clean),
        ("3. SfM+MVS Robust MAD Fusion       ", hr_sr_init[m:-m, m:-m]),
        ("4. SfM+MVS + MAP-BTV Super-Res     ", hr_sr_map[m:-m, m:-m]),
    ]:
        p, s, ss = compute_metrics(gt_c, img)
        print(f"{name} -> PSNR: {p:.2f} dB | SNR: {s:.2f} dB | SSIM: {ss:.4f}")
    print(f"Completed in {time.time() - t0:.2f} s.")