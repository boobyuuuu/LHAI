from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from morphology_templates import (
    FOV_DEG,
    NPIX_RESPONSE,
    NPIX_SUPER,
    PIXEL_SIZE_RESPONSE_DEG,
    PIXEL_SIZE_SUPER_DEG,
    diffusion_source,
    disk_source,
    downsample_sum_2x2,
    gaussian_source,
    make_grid,
    normalize_to_flux,
    point_source,
    shell_source,
)

SEED = 20260517
FLUX_COEFFS = np.geomspace(0.1, 10.0, 50, dtype=np.float64)
FLUX_ORDER = 1e-16
R39_GRID_DEG = np.linspace(0.1, 0.5, 20, dtype=np.float64)
DEC_CENTER_DEG = 22.0
CENTER_SIGMA_DEG = 0.75
CENTER_LIMIT_DEG = 1.6
TEMPLATE_FLUX = 1.0

SOURCE_TYPES = ("POINT", "GAUSSIAN", "DISK", "SHELL", "DIFFUSION")
EXTENDED_TYPES = ("GAUSSIAN", "DISK", "SHELL", "DIFFUSION")

RESPONSE_SETTINGS = {
    "detector": "KM2A",
    "energy_selection": ">25 TeV",
    "emin_log10_TeV": 1.4,
    "emax_log10_TeV": 3.4,
    "roi_dec_center_deg": DEC_CENTER_DEG,
    "roi_ra_center_sampling": "uniform_[0,360) in response generation; independent from source Cx/Cy",
    "epiv_TeV": 50.0,
    "alpha": 3.0,
    "flux_order": "1e-16 cm^-2 s^-1 TeV^-1",
    "flux_source": "response configuration F0 only; GT npy templates are unit-normalized",
}


def sample_source_center(rng: np.random.Generator) -> tuple[float, float]:
    while True:
        cx, cy = rng.normal(0.0, CENTER_SIGMA_DEG, size=2)
        if abs(cx) <= CENTER_LIMIT_DEG and abs(cy) <= CENTER_LIMIT_DEG:
            return float(cx), float(cy)


def disk_radius_from_r39(r39_deg: float) -> float:
    return float(r39_deg / math.sqrt(0.39))


def gaussian_sigma_from_r39(r39_deg: float) -> float:
    return float(r39_deg / math.sqrt(-2.0 * math.log(1.0 - 0.39)))


def random_params_for_source(source_type: str, r39_deg: float | None, rng: np.random.Generator) -> dict:
    if source_type == "POINT":
        return {"scale_parameter": "none", "scale_r39_deg": None, "point_marker_deg": 0.05}
    if source_type == "GAUSSIAN":
        epsilon = float(rng.uniform(0.0, 0.6))
        sigma_major = gaussian_sigma_from_r39(float(r39_deg))
        return {
            "scale_parameter": "r39_deg",
            "scale_r39_deg": float(r39_deg),
            "r39_deg": float(r39_deg),
            "sigma_major_deg": sigma_major,
            "epsilon": epsilon,
            "sigma_minor_deg": float(sigma_major * (1.0 - epsilon)),
            "position_angle_deg": float(rng.uniform(0.0, 180.0)),
        }
    if source_type == "DISK":
        radius = disk_radius_from_r39(float(r39_deg))
        return {"scale_parameter": "r39_deg", "scale_r39_deg": float(r39_deg), "r39_deg": float(r39_deg), "radius_deg": radius}
    if source_type == "SHELL":
        r_out = disk_radius_from_r39(float(r39_deg))
        thickness_fraction = float(rng.uniform(0.25, 0.55))
        r_in = float(max(0.05, r_out * (1.0 - thickness_fraction)))
        return {
            "scale_parameter": "r39_deg",
            "scale_r39_deg": float(r39_deg),
            "r39_deg": float(r39_deg),
            "r_in_deg": r_in,
            "r_out_deg": r_out,
            "shell_thickness_fraction": thickness_fraction,
        }
    if source_type == "DIFFUSION":
        r68 = float(r39_deg * rng.uniform(1.45, 1.85))
        return {"scale_parameter": "r39_deg", "scale_r39_deg": float(r39_deg), "r39_deg": float(r39_deg), "r68_deg": r68}
    raise ValueError(f"Unknown source type: {source_type}")


def morphology_params_for_builder(source_type: str, params: dict) -> dict:
    skip = {"scale_parameter", "scale_r39_deg", "point_marker_deg", "sigma_minor_deg", "shell_thickness_fraction"}
    if source_type == "POINT":
        return {}
    if source_type == "GAUSSIAN":
        return {key: value for key, value in params.items() if key not in skip and key != "r39_deg"}
    if source_type == "DISK":
        return {"radius_deg": params["radius_deg"]}
    if source_type == "SHELL":
        return {"r_in_deg": params["r_in_deg"], "r_out_deg": params["r_out_deg"]}
    if source_type == "DIFFUSION":
        return {"r39_deg": params["r39_deg"], "r68_deg": params["r68_deg"]}
    raise ValueError(f"Unknown source type: {source_type}")


def build_template(source_type: str, params: dict, cx_deg: float, cy_deg: float) -> tuple[np.ndarray, np.ndarray]:
    builders: dict[str, Callable[..., np.ndarray]] = {
        "POINT": point_source,
        "GAUSSIAN": gaussian_source,
        "DISK": disk_source,
        "SHELL": shell_source,
        "DIFFUSION": diffusion_source,
    }
    grid = make_grid(NPIX_SUPER, PIXEL_SIZE_SUPER_DEG, cx_deg=cx_deg, cy_deg=cy_deg)
    image_128 = builders[source_type](flux=TEMPLATE_FLUX, grid=grid, **morphology_params_for_builder(source_type, params))
    image_64 = normalize_to_flux(downsample_sum_2x2(image_128), TEMPLATE_FLUX)
    return image_128, image_64


def serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    outdir = Path(__file__).resolve().parent
    rng = np.random.default_rng(SEED)

    metadata = {
        "experiment": "Exp1_ParametricMorphologyReconstruction",
        "goal": "Single-source parametric morphology reconstruction confidence data over type, r39, flux, and source center position.",
        "global_settings": {
            "single_image_single_source": True,
            "seed": SEED,
            "template_normalization": "128 and 64 GT images are independently unit-normalized; LHAASO response flux is controlled only by F0 in ParInit.yaml.",
            "source_center": {
                "parameters": ["Cx_deg", "Cy_deg"],
                "sampling": "independent Gaussian N(0, 0.75 deg), truncated to |Cx|<=1.6 deg and |Cy|<=1.6 deg",
                "meaning": "source morphology center in the image coordinate system, not ROI RA/DEC",
            },
            "super_resolution_grid": {"npix": NPIX_SUPER, "pixel_size_deg": PIXEL_SIZE_SUPER_DEG, "fov_deg": FOV_DEG},
            "response_grid": {
                "npix": NPIX_RESPONSE,
                "pixel_size_deg": PIXEL_SIZE_RESPONSE_DEG,
                "fov_deg": FOV_DEG,
                "construction": "sum each 2x2 block from the 0.05 deg grid, then independently unit-normalize the 64 grid",
            },
            "response": RESPONSE_SETTINGS,
        },
        "parameter_strategy": {
            "source_types": list(SOURCE_TYPES),
            "flux_coefficients": FLUX_COEFFS,
            "flux_coefficients_rule": "np.geomspace(0.1, 10.0, 50), endpoints included; written to response F0 as coefficient × 1e-16",
            "r39_grid_deg": R39_GRID_DEG,
            "r39_grid_rule": "np.linspace(0.1, 0.5, 20), endpoints included; used as the confidence-map scale axis for GAUSSIAN/DISK/SHELL/DIFFUSION",
            "scale_definitions": {
                "POINT": "no intrinsic scale parameter; 50 samples over flux only",
                "GAUSSIAN": "scale_r39_deg is converted to sigma_major_deg using circular Gaussian containment r39 = sigma * sqrt(-2 ln(1-0.39)); epsilon and position angle are randomized",
                "DISK": "scale_r39_deg is converted to radius_deg using uniform disk containment r39 = radius * sqrt(0.39)",
                "SHELL": "scale_r39_deg is converted to r_out_deg using the same containment proxy as DISK; r_in_deg is randomized and recorded",
                "DIFFUSION": "scale_r39_deg is used directly as r39_deg; r68_deg is randomized and recorded",
            },
            "sample_order": "POINT iterates flux. Extended types iterate r39 grid first, then ascending log-spaced flux coefficients.",
        },
        "outputs": {},
        "samples": {},
    }

    for source_type in SOURCE_TYPES:
        images_128 = []
        images_64 = []
        samples = []
        scale_values = [None] if source_type == "POINT" else R39_GRID_DEG
        for size_index, r39_deg in enumerate(scale_values):
            for flux_index, flux_coeff in enumerate(FLUX_COEFFS):
                morphology_params = random_params_for_source(source_type, None if r39_deg is None else float(r39_deg), rng)
                cx_deg, cy_deg = sample_source_center(rng)
                image_128, image_64 = build_template(source_type, morphology_params, cx_deg, cy_deg)
                index = len(images_128)
                images_128.append(image_128)
                images_64.append(image_64)
                samples.append(
                    {
                        "index": index,
                        "source_type": source_type,
                        "size_index": size_index if source_type != "POINT" else None,
                        "flux_index": flux_index,
                        "flux_coefficient": float(flux_coeff),
                        "flux_order": "1e-16",
                        "response_F0": float(flux_coeff),
                        "response_flux_unit": "1e-16 cm^-2 s^-1 TeV^-1",
                        "template_sum_128": TEMPLATE_FLUX,
                        "template_sum_64": TEMPLATE_FLUX,
                        "Cx_deg": cx_deg,
                        "Cy_deg": cy_deg,
                        "roi_dec_center_deg": DEC_CENTER_DEG,
                        "scale_parameter": morphology_params["scale_parameter"],
                        "scale_r39_deg": morphology_params["scale_r39_deg"],
                        "morphology_parameters": morphology_params,
                    }
                )

        array_128 = np.stack(images_128).astype(np.float32)
        array_64 = np.stack(images_64).astype(np.float32)
        sample_count = array_64.shape[0]
        npy_128_name = f"EXP1{source_type}_{sample_count}_128_GT.npy"
        npy_64_name = f"EXP1{source_type}_{sample_count}_64_GT.npy"
        np.save(outdir / npy_128_name, array_128)
        np.save(outdir / npy_64_name, array_64)
        metadata["outputs"][source_type] = {
            "gt_128_file": npy_128_name,
            "gt_64_file": npy_64_name,
            "response_input_file": npy_64_name,
            "expected_response_file": f"EXP1{source_type}_{sample_count}_64_RESPONSE.npy",
            "shape_128": list(array_128.shape),
            "shape_64": list(array_64.shape),
            "dtype": "float32",
        }
        metadata["samples"][source_type] = samples

    with (outdir / "exp1_parameters.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=serializable)

    print(f"Wrote Experiment 1 GT arrays and exp1_parameters.json to {outdir}")
    for source_type, output in metadata["outputs"].items():
        print(
            f"{source_type}: {output['gt_128_file']} shape={tuple(output['shape_128'])}; "
            f"{output['gt_64_file']} shape={tuple(output['shape_64'])}"
        )


if __name__ == "__main__":
    main()
