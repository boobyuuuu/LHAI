import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


SIM_RES = 128
FOV_DEG = 6.4
PIXEL_SCALE_DEG = FOV_DEG / SIM_RES
INFERNO = LinearSegmentedColormap.from_list(
    "compact_core_inferno",
    ["#000000", "#2D005C", "#8B1A5B", "#E85D04", "#FFE66D", "#FFFFFF"],
)
DATASET_OUTPUT = "Type_A_Compact_100_128_GT.npy"


def get_log_uniform_intensity(vmin, vmax, rng):
    return 10 ** rng.uniform(np.log10(vmin), np.log10(vmax))


def make_coordinate_grid(size=SIM_RES, fov=FOV_DEG):
    pixel_size = fov / size
    axis = np.linspace(-fov / 2, fov / 2, size, endpoint=False) + pixel_size / 2
    return np.meshgrid(axis.astype(np.float32), axis.astype(np.float32))


def get_gaussian_center(center_mu=0.0, center_sigma=1.5, max_offset=3.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    while True:
        cx = rng.normal(center_mu, center_sigma)
        cy = rng.normal(center_mu, center_sigma)
        if abs(cx) < max_offset and abs(cy) < max_offset:
            return cx, cy


def make_point(x, y, cx, cy, intensity):
    image = np.zeros_like(x, dtype=np.float32)
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    min_idx = np.unravel_index(np.argmin(dist_sq, axis=None), dist_sq.shape)
    image[min_idx] = intensity
    return image


def make_gaussian(x, y, cx, cy, sigma, intensity):
    image = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma)))
    total = image.sum(dtype=np.float64)
    if total > 0:
        image = image / total * intensity
    return image.astype(np.float32, copy=False)


def simulate_compact_core(
    source_kind="compact",
    center_mu=0.0,
    center_sigma=1.5,
    max_offset=3.2,
    intensity=10.0,
    sigma=0.05,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
):
    rng = np.random.default_rng(seed)
    x, y = make_coordinate_grid(size=size, fov=fov)
    cx, cy = get_gaussian_center(
        center_mu=center_mu,
        center_sigma=center_sigma,
        max_offset=max_offset,
        rng=rng,
    )

    if source_kind == "point":
        image = make_point(x, y, cx, cy, intensity)
        params = {"type": "A_Point", "cx": cx, "cy": cy, "sigma": 0.0, "intensity": intensity}
    elif source_kind == "compact":
        image = make_gaussian(x, y, cx, cy, sigma, intensity)
        params = {"type": "A_Compact", "cx": cx, "cy": cy, "sigma": sigma, "intensity": intensity}
    else:
        raise ValueError("source_kind must be 'point' or 'compact'")

    return image, params


def simulate_random_type_a(
    center_mu=0.0,
    center_sigma=1.5,
    max_offset=3.2,
    intensity_range=(1, 100),
    sigma_range=(0.01, 0.1),
    point_fraction=0.5,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
):
    rng = np.random.default_rng(seed)
    source_kind = "point" if rng.random() < point_fraction else "compact"
    intensity = get_log_uniform_intensity(*intensity_range, rng=rng)
    sigma = rng.uniform(*sigma_range)

    x, y = make_coordinate_grid(size=size, fov=fov)
    cx, cy = get_gaussian_center(
        center_mu=center_mu,
        center_sigma=center_sigma,
        max_offset=max_offset,
        rng=rng,
    )

    if source_kind == "point":
        image = make_point(x, y, cx, cy, intensity)
        params = {"type": "A_Point", "cx": cx, "cy": cy, "sigma": 0.0, "intensity": intensity}
    else:
        image = make_gaussian(x, y, cx, cy, sigma, intensity)
        params = {"type": "A_Compact", "cx": cx, "cy": cy, "sigma": sigma, "intensity": intensity}

    return image, params


def generate_type_a_compact_dataset(
    count=100,
    center_mu=0.0,
    center_sigma=0.05,
    max_offset=3.2,
    intensity=10.0,
    intensity_range=(1, 100),
    sigma=0.05,
    sigma_range=(0.01, 0.1),
    point_fraction=0.5,
    random=False,
    size=SIM_RES,
    fov=FOV_DEG,
    output=DATASET_OUTPUT,
    seed=None,
):
    rng = np.random.default_rng(seed)
    x, y = make_coordinate_grid(size=size, fov=fov)
    dataset = np.empty((count, size, size), dtype=np.float32)

    for i in range(count):
        cx, cy = get_gaussian_center(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            rng=rng,
        )

        if random:
            source_kind = "point" if rng.random() < point_fraction else "compact"
            sample_intensity = get_log_uniform_intensity(*intensity_range, rng=rng)
            sample_sigma = rng.uniform(*sigma_range)
        else:
            source_kind = "compact"
            sample_intensity = intensity
            sample_sigma = sigma

        if source_kind == "point":
            dataset[i] = make_point(x, y, cx, cy, sample_intensity)
        else:
            dataset[i] = make_gaussian(x, y, cx, cy, sample_sigma, sample_intensity)

    np.save(output, dataset)
    return dataset


def show_type_a_core(
    source_kind="compact",
    random=False,
    center_mu=0.0,
    center_sigma=1.5,
    max_offset=3.2,
    intensity=10.0,
    intensity_min=1.0,
    intensity_max=100.0,
    sigma=0.05,
    sigma_min=0.01,
    sigma_max=0.1,
    point_fraction=0.5,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
    save=None,
):
    if random:
        image, params = simulate_random_type_a(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity_range=(intensity_min, intensity_max),
            sigma_range=(sigma_min, sigma_max),
            point_fraction=point_fraction,
            size=size,
            fov=fov,
            seed=seed,
        )
    else:
        image, params = simulate_compact_core(
            source_kind=source_kind,
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity=intensity,
            sigma=sigma,
            size=size,
            fov=fov,
            seed=seed,
        )

    total_flux = float(image.sum())
    peak_flux = float(image.max())

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    image_plot = ax.imshow(
        image,
        origin="lower",
        cmap=INFERNO,
        extent=[-fov / 2, fov / 2, -fov / 2, fov / 2],
    )
    ax.set_title("Type A Compact Source Preview")
    ax.set_xlabel("RA offset (deg)")
    ax.set_ylabel("Dec offset (deg)")

    ax.text(
        0.02,
        0.98,
        f"{params['type']}\n"
        f"cx={params['cx']:.3f}°, cy={params['cy']:.3f}°\n"
        f"sigma={params['sigma']:.3f}°\n"
        f"Flux={total_flux:.3f}, Peak={peak_flux:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 4},
    )
    fig.colorbar(image_plot, ax=ax, fraction=0.046, pad=0.04, label="Flux")

    if save:
        fig.savefig(save, dpi=200)
    plt.show()

    return image, params


def parse_args():
    parser = argparse.ArgumentParser(description="Preview Type-A point/compact-core source generation.")
    parser.add_argument("--source-kind", choices=["point", "compact"], default="compact", help="Single-image morphology")
    parser.add_argument("--random", action="store_true", help="Randomly choose point/compact and sample intensity/sigma ranges")
    parser.add_argument("--center-mu", type=float, default=0.0, help="Gaussian center distribution mean in degrees")
    parser.add_argument("--center-sigma", type=float, default=1.5, help="Gaussian center distribution sigma in degrees")
    parser.add_argument("--max-offset", type=float, default=3.2, help="Maximum absolute center offset in degrees")
    parser.add_argument("--intensity", type=float, default=10.0, help="Total source intensity for non-random preview")
    parser.add_argument("--intensity-min", type=float, default=1.0, help="Random preview minimum total intensity")
    parser.add_argument("--intensity-max", type=float, default=100.0, help="Random preview maximum total intensity")
    parser.add_argument("--sigma", type=float, default=0.05, help="Compact Gaussian sigma in degrees")
    parser.add_argument("--sigma-min", type=float, default=0.01, help="Random preview minimum compact sigma in degrees")
    parser.add_argument("--sigma-max", type=float, default=0.1, help="Random preview maximum compact sigma in degrees")
    parser.add_argument("--point-fraction", type=float, default=0.5, help="Random preview probability of drawing a point source")
    parser.add_argument("--size", type=int, default=SIM_RES, help="Image width and height in pixels")
    parser.add_argument("--fov", type=float, default=FOV_DEG, help="Field of view in degrees")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible preview")
    parser.add_argument("--save", default=None, help="Optional path to save the plotted image")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate the Type-A compact dataset")
    parser.add_argument("--dataset-count", type=int, default=100, help="Number of compact images in the dataset")
    parser.add_argument("--dataset-output", default=DATASET_OUTPUT, help="Output .npy path for the compact dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.generate_dataset:
        data = generate_type_a_compact_dataset(
            count=args.dataset_count,
            center_mu=args.center_mu,
            center_sigma=args.center_sigma,
            max_offset=args.max_offset,
            intensity=args.intensity,
            intensity_range=(args.intensity_min, args.intensity_max),
            sigma=args.sigma,
            sigma_range=(args.sigma_min, args.sigma_max),
            point_fraction=args.point_fraction,
            random=args.random,
            size=args.size,
            fov=args.fov,
            output=args.dataset_output,
            seed=args.seed,
        )
        flux = data.sum(axis=(1, 2))
        print(
            f"Saved {args.dataset_output}: shape={data.shape}, dtype={data.dtype}, "
            f"min={data.min():.6e}, max={data.max():.6e}"
        )
        print(
            f"Flux sum: min={flux.min():.6f}, max={flux.max():.6f}, mean={flux.mean():.6f}"
        )
    else:
        show_type_a_core(
            source_kind=args.source_kind,
            random=args.random,
            center_mu=args.center_mu,
            center_sigma=args.center_sigma,
            max_offset=args.max_offset,
            intensity=args.intensity,
            intensity_min=args.intensity_min,
            intensity_max=args.intensity_max,
            sigma=args.sigma,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            point_fraction=args.point_fraction,
            size=args.size,
            fov=args.fov,
            seed=args.seed,
            save=args.save,
        )
