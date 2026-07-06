from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NPIX_RESPONSE = 64
PIXEL_SIZE_RESPONSE_DEG = 0.1
NPIX_SUPER = 128
PIXEL_SIZE_SUPER_DEG = 0.05
FOV_DEG = NPIX_RESPONSE * PIXEL_SIZE_RESPONSE_DEG


@dataclass(frozen=True)
class Grid:
    npix: int
    pixel_size_deg: float
    x_deg: np.ndarray
    y_deg: np.ndarray
    r_deg: np.ndarray


def make_grid(npix: int, pixel_size_deg: float, cx_deg: float = 0.0, cy_deg: float = 0.0) -> Grid:
    axis = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.0) * pixel_size_deg
    x_deg, y_deg = np.meshgrid(axis, axis)
    r_deg = np.hypot(x_deg - cx_deg, y_deg - cy_deg)
    return Grid(npix=npix, pixel_size_deg=pixel_size_deg, x_deg=x_deg, y_deg=y_deg, r_deg=r_deg)


def normalize_to_flux(template: np.ndarray, flux: float) -> np.ndarray:
    image = np.asarray(template, dtype=np.float64)
    image = np.clip(image, 0.0, None)
    total = float(image.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Template must have a positive finite sum.")
    return (image / total * float(flux)).astype(np.float32)


def downsample_sum_2x2(image: np.ndarray) -> np.ndarray:
    if image.shape != (NPIX_SUPER, NPIX_SUPER):
        raise ValueError(f"Expected ({NPIX_SUPER}, {NPIX_SUPER}), got {image.shape}")
    return image.reshape(NPIX_RESPONSE, 2, NPIX_RESPONSE, 2).sum(axis=(1, 3)).astype(np.float32)


def point_source(flux: float, grid: Grid) -> np.ndarray:
    image = np.zeros((grid.npix, grid.npix), dtype=np.float64)
    nearest = np.unravel_index(np.argmin(grid.r_deg), grid.r_deg.shape)
    image[nearest] = 1.0
    return normalize_to_flux(image, flux)


def gaussian_source(
    flux: float,
    sigma_major_deg: float,
    epsilon: float,
    position_angle_deg: float,
    grid: Grid,
) -> np.ndarray:
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1).")
    sigma_minor_deg = sigma_major_deg * (1.0 - epsilon)
    if sigma_major_deg <= 0.0 or sigma_minor_deg <= 0.0:
        raise ValueError("Gaussian sigmas must be positive.")
    theta = math.radians(position_angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    cx = grid.x_deg - grid.x_deg.flat[np.argmin(grid.r_deg)]
    cy = grid.y_deg - grid.y_deg.flat[np.argmin(grid.r_deg)]
    x_rot = cx * cos_t + cy * sin_t
    y_rot = -cx * sin_t + cy * cos_t
    image = np.exp(-0.5 * ((x_rot / sigma_major_deg) ** 2 + (y_rot / sigma_minor_deg) ** 2))
    return normalize_to_flux(image, flux)


def disk_source(flux: float, radius_deg: float, grid: Grid) -> np.ndarray:
    if radius_deg <= 0.0:
        raise ValueError("Disk radius must be positive.")
    edge_width = grid.pixel_size_deg
    image = 1.0 / (1.0 + np.exp((grid.r_deg - radius_deg) / edge_width))
    return normalize_to_flux(image, flux)


def shell_source(flux: float, r_in_deg: float, r_out_deg: float, grid: Grid) -> np.ndarray:
    if r_in_deg < 0.0 or r_out_deg <= r_in_deg:
        raise ValueError("Shell requires 0 <= r_in_deg < r_out_deg.")
    edge_width = grid.pixel_size_deg
    outer = 1.0 / (1.0 + np.exp((grid.r_deg - r_out_deg) / edge_width))
    inner = 1.0 / (1.0 + np.exp((grid.r_deg - r_in_deg) / edge_width))
    image = np.clip(outer - inner, 0.0, None)
    return normalize_to_flux(image, flux)


def diffusion_source(flux: float, r39_deg: float, r68_deg: float, grid: Grid) -> np.ndarray:
    if r39_deg <= 0.0 or r68_deg <= r39_deg:
        raise ValueError("Diffusion requires 0 < r39_deg < r68_deg.")
    sigma_from_r68 = r68_deg / math.sqrt(-2.0 * math.log(1.0 - 0.68))
    sigma_from_r39 = r39_deg / math.sqrt(-2.0 * math.log(1.0 - 0.39))
    sigma_deg = 0.5 * (sigma_from_r39 + sigma_from_r68)
    core = np.exp(-0.5 * (grid.r_deg / sigma_deg) ** 2)
    tail = np.exp(-grid.r_deg / max(r68_deg, grid.pixel_size_deg))
    image = 0.75 * core + 0.25 * tail
    return normalize_to_flux(image, flux)
