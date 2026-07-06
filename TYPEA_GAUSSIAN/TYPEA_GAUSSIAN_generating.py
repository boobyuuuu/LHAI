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
SOURCE_TYPES = ("perfect", "elliptical", "perftextured", "elliptextured")


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


def normalize_intensity(image, intensity):
    total = image.sum(dtype=np.float64)
    if total > 0:
        image = image / total * intensity
    return image.astype(np.float32, copy=False)


def make_gaussian(x, y, cx, cy, sigma):
    return np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma)))


def make_elliptical_gaussian(x, y, cx, cy, sigma, axis_ratio, theta):
    dx = x - cx
    dy = y - cy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    sigma_x = sigma
    sigma_y = sigma * axis_ratio
    return np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))


def make_fractal_noise(size=SIM_RES, beta=3.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    freq = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freq, freq)
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0

    amplitude = f ** (-beta)
    amplitude[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, (size, size))
    spectrum = amplitude * np.exp(1j * phase)

    noise = np.real(np.fft.ifft2(spectrum))
    noise = noise - noise.min()
    max_value = noise.max()
    if max_value > 0:
        noise = noise / max_value
    return noise.astype(np.float32, copy=False)


def apply_bow_shock_modulation(x, y, image, cx, cy, phi, compression=0.65, tail_strength=1.2, transition=0.12):
    dx = x - cx
    dy = y - cy
    s = dx * np.cos(phi) + dy * np.sin(phi)

    head_gate = 1.0 / (1.0 + np.exp(s / transition))
    tail = np.exp(-np.maximum(s, 0.0) / tail_strength)
    modulation = compression + (1.0 - compression) * head_gate + (1.0 - compression) * tail

    return image * modulation


def apply_turbulence_modulation(image, alpha=0.2, beta=3.0, rng=None):
    noise = make_fractal_noise(size=image.shape[0], beta=beta, rng=rng)
    modulation = 1.0 - alpha + 2.0 * alpha * noise
    return image * modulation


def simulate_perfect_gaussian(
    center_mu=0.0,
    center_sigma=1.5,
    max_offset=3.2,
    intensity=10.0,
    sigma=0.05,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(seed)
    x, y = make_coordinate_grid(size=size, fov=fov)
    cx, cy = get_gaussian_center(
        center_mu=center_mu,
        center_sigma=center_sigma,
        max_offset=max_offset,
        rng=rng,
    )
    image = make_gaussian(x, y, cx, cy, sigma)
    image = normalize_intensity(image, intensity)
    params = {
        "type": "A_PerfectGaussian",
        "cx": cx,
        "cy": cy,
        "sigma": sigma,
        "axis_ratio": 1.0,
        "theta": 0.0,
        "intensity": intensity,
    }
    return image, params


def simulate_elliptical_gaussian(
    center_mu=0.0,
    center_sigma=1.5,
    max_offset=3.2,
    intensity=10.0,
    sigma=0.05,
    axis_ratio=0.7,
    theta=None,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(seed)
    x, y = make_coordinate_grid(size=size, fov=fov)
    cx, cy = get_gaussian_center(
        center_mu=center_mu,
        center_sigma=center_sigma,
        max_offset=max_offset,
        rng=rng,
    )
    if theta is None:
        theta = rng.uniform(0, np.pi)

    image = make_elliptical_gaussian(x, y, cx, cy, sigma, axis_ratio, theta)
    image = normalize_intensity(image, intensity)
    params = {
        "type": "A_EllipticalGaussian",
        "cx": cx,
        "cy": cy,
        "sigma": sigma,
        "axis_ratio": axis_ratio,
        "theta": theta,
        "intensity": intensity,
    }
    return image, params


def simulate_textured_gaussian(
    base_kind="elliptical",
    center_mu=0.0,
    center_sigma=1.5,
    max_offset=3.2,
    intensity=10.0,
    sigma=0.05,
    axis_ratio=0.7,
    theta=None,
    bow_phi=None,
    compression=0.65,
    tail_strength=1.2,
    transition=0.12,
    turbulence_alpha=0.2,
    turbulence_beta=3.0,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(seed)
    x, y = make_coordinate_grid(size=size, fov=fov)
    cx, cy = get_gaussian_center(
        center_mu=center_mu,
        center_sigma=center_sigma,
        max_offset=max_offset,
        rng=rng,
    )
    if theta is None:
        theta = rng.uniform(0, np.pi)
    if bow_phi is None:
        bow_phi = rng.uniform(0, 2 * np.pi)

    if base_kind == "perfect":
        image = make_gaussian(x, y, cx, cy, sigma)
        params_type = "A_PerfTexturedGaussian"
        params_axis_ratio = 1.0
        params_theta = 0.0
    elif base_kind == "elliptical":
        image = make_elliptical_gaussian(x, y, cx, cy, sigma, axis_ratio, theta)
        params_type = "A_EllipTexturedGaussian"
        params_axis_ratio = axis_ratio
        params_theta = theta
    else:
        raise ValueError("base_kind must be 'perfect' or 'elliptical'")

    image = apply_bow_shock_modulation(
        x=x,
        y=y,
        image=image,
        cx=cx,
        cy=cy,
        phi=bow_phi,
        compression=compression,
        tail_strength=tail_strength,
        transition=transition,
    )
    image = apply_turbulence_modulation(
        image=image,
        alpha=turbulence_alpha,
        beta=turbulence_beta,
        rng=rng,
    )
    image = normalize_intensity(image, intensity)
    params = {
        "type": params_type,
        "cx": cx,
        "cy": cy,
        "sigma": sigma,
        "axis_ratio": params_axis_ratio,
        "theta": params_theta,
        "bow_phi": bow_phi,
        "compression": compression,
        "tail_strength": tail_strength,
        "transition": transition,
        "turbulence_alpha": turbulence_alpha,
        "turbulence_beta": turbulence_beta,
        "intensity": intensity,
    }
    return image, params


def choose_source_type(perfect_fraction, elliptical_fraction, perftextured_fraction, elliptextured_fraction, rng):
    weights = np.array(
        [perfect_fraction, elliptical_fraction, perftextured_fraction, elliptextured_fraction],
        dtype=np.float64,
    )
    if np.any(weights < 0):
        raise ValueError("Source-type fractions must be non-negative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("At least one source-type fraction must be positive")
    weights = weights / total
    return rng.choice(SOURCE_TYPES, p=weights)


def simulate_type_a_gaussian(
    source_kind="perfect",
    random=False,
    center_mu=0.0,
    center_sigma=1.5,
    max_offset=3.2,
    intensity=10.0,
    intensity_range=(1, 100),
    sigma=0.05,
    sigma_range=(0.01, 0.05),
    axis_ratio=0.7,
    axis_ratio_range=(0.45, 1.0),
    compression=0.65,
    compression_range=(0.35, 0.9),
    tail_strength=1.2,
    tail_strength_range=(0.5, 2.0),
    transition=0.12,
    transition_range=(0.05, 0.25),
    turbulence_alpha=0.2,
    turbulence_alpha_range=(0.05, 0.3),
    turbulence_beta=3.0,
    perfect_fraction=1.0,
    elliptical_fraction=1.0,
    perftextured_fraction=1.0,
    elliptextured_fraction=1.0,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(seed)

    if random:
        source_kind = choose_source_type(
            perfect_fraction,
            elliptical_fraction,
            perftextured_fraction,
            elliptextured_fraction,
            rng,
        )
        intensity = get_log_uniform_intensity(*intensity_range, rng=rng)
        sigma = rng.uniform(*sigma_range)
        axis_ratio = rng.uniform(*axis_ratio_range)
        compression = rng.uniform(*compression_range)
        tail_strength = rng.uniform(*tail_strength_range)
        transition = rng.uniform(*transition_range)
        turbulence_alpha = rng.uniform(*turbulence_alpha_range)

    if source_kind == "perfect":
        return simulate_perfect_gaussian(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity=intensity,
            sigma=sigma,
            size=size,
            fov=fov,
            rng=rng,
        )
    if source_kind == "elliptical":
        return simulate_elliptical_gaussian(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity=intensity,
            sigma=sigma,
            axis_ratio=axis_ratio,
            theta=None if random else 0.0,
            size=size,
            fov=fov,
            rng=rng,
        )
    if source_kind == "perftextured":
        return simulate_textured_gaussian(
            base_kind="perfect",
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity=intensity,
            sigma=sigma,
            axis_ratio=axis_ratio,
            theta=None if random else 0.0,
            bow_phi=None if random else 0.0,
            compression=compression,
            tail_strength=tail_strength,
            transition=transition,
            turbulence_alpha=turbulence_alpha,
            turbulence_beta=turbulence_beta,
            size=size,
            fov=fov,
            rng=rng,
        )
    if source_kind == "elliptextured":
        return simulate_textured_gaussian(
            base_kind="elliptical",
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity=intensity,
            sigma=sigma,
            axis_ratio=axis_ratio,
            theta=None if random else 0.0,
            bow_phi=None if random else 0.0,
            compression=compression,
            tail_strength=tail_strength,
            transition=transition,
            turbulence_alpha=turbulence_alpha,
            turbulence_beta=turbulence_beta,
            size=size,
            fov=fov,
            rng=rng,
        )
    raise ValueError(f"source_kind must be one of {SOURCE_TYPES}")


def generate_type_a_compact_dataset(
    count=100,
    center_mu=0.0,
    center_sigma=0.05,
    max_offset=3.2,
    intensity=10.0,
    intensity_range=(1, 100),
    sigma=0.05,
    sigma_range=(0.01, 0.05),
    axis_ratio=0.7,
    axis_ratio_range=(0.45, 1.0),
    compression=0.65,
    compression_range=(0.35, 0.9),
    tail_strength=1.2,
    tail_strength_range=(0.5, 2.0),
    transition=0.12,
    transition_range=(0.05, 0.25),
    turbulence_alpha=0.2,
    turbulence_alpha_range=(0.05, 0.3),
    turbulence_beta=3.0,
    perfect_fraction=1.0,
    elliptical_fraction=0.0,
    perftextured_fraction=0.0,
    elliptextured_fraction=0.0,
    random=False,
    size=SIM_RES,
    fov=FOV_DEG,
    output=DATASET_OUTPUT,
    seed=None,
):
    rng = np.random.default_rng(seed)
    dataset = np.empty((count, size, size), dtype=np.float32)

    for i in range(count):
        source_kind = "perfect"
        sample_intensity = intensity
        sample_sigma = sigma
        sample_axis_ratio = axis_ratio
        sample_compression = compression
        sample_tail_strength = tail_strength
        sample_transition = transition
        sample_turbulence_alpha = turbulence_alpha
        sample_theta = 0.0
        sample_bow_phi = 0.0

        if random:
            source_kind = choose_source_type(
                perfect_fraction,
                elliptical_fraction,
                perftextured_fraction,
                elliptextured_fraction,
                rng,
            )
            sample_intensity = get_log_uniform_intensity(*intensity_range, rng=rng)
            sample_sigma = rng.uniform(*sigma_range)
            sample_axis_ratio = rng.uniform(*axis_ratio_range)
            sample_compression = rng.uniform(*compression_range)
            sample_tail_strength = rng.uniform(*tail_strength_range)
            sample_transition = rng.uniform(*transition_range)
            sample_turbulence_alpha = rng.uniform(*turbulence_alpha_range)
            sample_theta = rng.uniform(0, np.pi)
            sample_bow_phi = rng.uniform(0, 2 * np.pi)

        if source_kind == "perfect":
            image, _ = simulate_perfect_gaussian(
                center_mu=center_mu,
                center_sigma=center_sigma,
                max_offset=max_offset,
                intensity=sample_intensity,
                sigma=sample_sigma,
                size=size,
                fov=fov,
                rng=rng,
            )
        elif source_kind == "elliptical":
            image, _ = simulate_elliptical_gaussian(
                center_mu=center_mu,
                center_sigma=center_sigma,
                max_offset=max_offset,
                intensity=sample_intensity,
                sigma=sample_sigma,
                axis_ratio=sample_axis_ratio,
                theta=sample_theta,
                size=size,
                fov=fov,
                rng=rng,
            )
        elif source_kind == "perftextured":
            image, _ = simulate_textured_gaussian(
                base_kind="perfect",
                center_mu=center_mu,
                center_sigma=center_sigma,
                max_offset=max_offset,
                intensity=sample_intensity,
                sigma=sample_sigma,
                axis_ratio=sample_axis_ratio,
                theta=sample_theta,
                bow_phi=sample_bow_phi,
                compression=sample_compression,
                tail_strength=sample_tail_strength,
                transition=sample_transition,
                turbulence_alpha=sample_turbulence_alpha,
                turbulence_beta=turbulence_beta,
                size=size,
                fov=fov,
                rng=rng,
            )
        else:
            image, _ = simulate_textured_gaussian(
                base_kind="elliptical",
                center_mu=center_mu,
                center_sigma=center_sigma,
                max_offset=max_offset,
                intensity=sample_intensity,
                sigma=sample_sigma,
                axis_ratio=sample_axis_ratio,
                theta=sample_theta,
                bow_phi=sample_bow_phi,
                compression=sample_compression,
                tail_strength=sample_tail_strength,
                transition=sample_transition,
                turbulence_alpha=sample_turbulence_alpha,
                turbulence_beta=turbulence_beta,
                size=size,
                fov=fov,
                rng=rng,
            )
        dataset[i] = image

    np.save(output, dataset)
    return dataset


def show_type_a_core(
    source_kind="perfect",
    random=False,
    center_mu=0.0,
    center_sigma=1.5,
    max_offset=3.2,
    intensity=10.0,
    intensity_min=1.0,
    intensity_max=100.0,
    sigma=0.05,
    sigma_min=0.01,
    sigma_max=0.05,
    axis_ratio=0.7,
    axis_ratio_min=0.45,
    axis_ratio_max=1.0,
    compression=0.65,
    compression_min=0.35,
    compression_max=0.9,
    tail_strength=1.2,
    tail_strength_min=0.5,
    tail_strength_max=2.0,
    transition=0.12,
    transition_min=0.05,
    transition_max=0.25,
    turbulence_alpha=0.2,
    turbulence_alpha_min=0.05,
    turbulence_alpha_max=0.3,
    turbulence_beta=3.0,
    perfect_fraction=1.0,
    elliptical_fraction=1.0,
    perftextured_fraction=1.0,
    elliptextured_fraction=1.0,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
    save=None,
):
    image, params = simulate_type_a_gaussian(
        source_kind=source_kind,
        random=random,
        center_mu=center_mu,
        center_sigma=center_sigma,
        max_offset=max_offset,
        intensity=intensity,
        intensity_range=(intensity_min, intensity_max),
        sigma=sigma,
        sigma_range=(sigma_min, sigma_max),
        axis_ratio=axis_ratio,
        axis_ratio_range=(axis_ratio_min, axis_ratio_max),
        compression=compression,
        compression_range=(compression_min, compression_max),
        tail_strength=tail_strength,
        tail_strength_range=(tail_strength_min, tail_strength_max),
        transition=transition,
        transition_range=(transition_min, transition_max),
        turbulence_alpha=turbulence_alpha,
        turbulence_alpha_range=(turbulence_alpha_min, turbulence_alpha_max),
        turbulence_beta=turbulence_beta,
        perfect_fraction=perfect_fraction,
        elliptical_fraction=elliptical_fraction,
        perftextured_fraction=perftextured_fraction,
        elliptextured_fraction=elliptextured_fraction,
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
    ax.set_title("Type A Gaussian Source Preview")
    ax.set_xlabel("RA offset (deg)")
    ax.set_ylabel("Dec offset (deg)")

    text = (
        f"{params['type']}\n"
        f"cx={params['cx']:.3f}°, cy={params['cy']:.3f}°\n"
        f"sigma={params['sigma']:.3f}°, q={params['axis_ratio']:.2f}\n"
        f"Flux={total_flux:.3f}, Peak={peak_flux:.3f}"
    )
    if params["type"] in ("A_PerfTexturedGaussian", "A_EllipTexturedGaussian"):
        text += f"\nalpha={params['turbulence_alpha']:.2f}, comp={params['compression']:.2f}"

    ax.text(
        0.02,
        0.98,
        text,
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
    parser = argparse.ArgumentParser(description="Preview or generate Type-A Gaussian source morphologies.")
    parser.add_argument(
        "--source-kind",
        choices=SOURCE_TYPES,
        default="perfect",
        help="Single-image morphology used when --random is not set",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly choose the Gaussian subtype and sample physical parameter ranges",
    )
    parser.add_argument("--center-mu", type=float, default=0.0, help="Gaussian center distribution mean in degrees")
    parser.add_argument("--center-sigma", type=float, default=1.0, help="Gaussian center distribution sigma in degrees")
    parser.add_argument("--max-offset", type=float, default=1.0, help="Maximum absolute center offset in degrees")
    parser.add_argument("--intensity", type=float, default=10.0, help="Total source intensity for non-random mode")
    parser.add_argument("--intensity-min", type=float, default=1.0, help="Random mode minimum total intensity")
    parser.add_argument("--intensity-max", type=float, default=100.0, help="Random mode maximum total intensity")
    parser.add_argument("--sigma", type=float, default=0.1, help="Gaussian sigma in degrees for non-random mode")
    parser.add_argument("--sigma-min", type=float, default=0.3, help="Random mode minimum sigma in degrees")
    parser.add_argument("--sigma-max", type=float, default=1.0, help="Random mode maximum sigma in degrees")
    parser.add_argument("--axis-ratio", type=float, default=0.7, help="Minor/major axis ratio for elliptical/textured Gaussian")
    parser.add_argument("--axis-ratio-min", type=float, default=0.20, help="Random mode minimum axis ratio")
    parser.add_argument("--axis-ratio-max", type=float, default=1.0, help="Random mode maximum axis ratio")
    parser.add_argument("--compression", type=float, default=0.65, help="Textured Gaussian compression floor")
    parser.add_argument("--compression-min", type=float, default=0.35, help="Random mode minimum compression")
    parser.add_argument("--compression-max", type=float, default=0.9, help="Random mode maximum compression")
    parser.add_argument("--tail-strength", type=float, default=1.2, help="Textured Gaussian tail decay scale in degrees")
    parser.add_argument("--tail-strength-min", type=float, default=0.5, help="Random mode minimum tail strength")
    parser.add_argument("--tail-strength-max", type=float, default=2.0, help="Random mode maximum tail strength")
    parser.add_argument("--transition", type=float, default=0.12, help="Textured Gaussian sigmoid transition width in degrees")
    parser.add_argument("--transition-min", type=float, default=0.05, help="Random mode minimum transition width")
    parser.add_argument("--transition-max", type=float, default=0.25, help="Random mode maximum transition width")
    parser.add_argument("--turbulence-alpha", type=float, default=0.2, help="Textured Gaussian turbulence modulation amplitude")
    parser.add_argument("--turbulence-alpha-min", type=float, default=0.05, help="Random mode minimum turbulence alpha")
    parser.add_argument("--turbulence-alpha-max", type=float, default=0.3, help="Random mode maximum turbulence alpha")
    parser.add_argument("--turbulence-beta", type=float, default=3.0, help="Fractal turbulence spectral index")
    parser.add_argument("--perfect-fraction", type=float, default=0.6, help="Dataset/random-preview relative fraction for perfect Gaussian")
    parser.add_argument("--elliptical-fraction", type=float, default=0.4, help="Dataset/random-preview relative fraction for elliptical Gaussian")
    parser.add_argument("--perftextured-fraction", type=float, default=0.0, help="Dataset/random-preview relative fraction for perfect-base textured Gaussian")
    parser.add_argument("--elliptextured-fraction", type=float, default=0.0, help="Dataset/random-preview relative fraction for elliptical-base textured Gaussian")
    parser.add_argument("--size", type=int, default=SIM_RES, help="Image width and height in pixels")
    parser.add_argument("--fov", type=float, default=FOV_DEG, help="Field of view in degrees")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation")
    parser.add_argument("--save", default=None, help="Optional path to save the plotted image")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate the Type-A Gaussian dataset")
    parser.add_argument("--dataset-count", type=int, default=100, help="Number of images in the dataset")
    parser.add_argument("--dataset-output", default=DATASET_OUTPUT, help="Output .npy path for the dataset")
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
            axis_ratio=args.axis_ratio,
            axis_ratio_range=(args.axis_ratio_min, args.axis_ratio_max),
            compression=args.compression,
            compression_range=(args.compression_min, args.compression_max),
            tail_strength=args.tail_strength,
            tail_strength_range=(args.tail_strength_min, args.tail_strength_max),
            transition=args.transition,
            transition_range=(args.transition_min, args.transition_max),
            turbulence_alpha=args.turbulence_alpha,
            turbulence_alpha_range=(args.turbulence_alpha_min, args.turbulence_alpha_max),
            turbulence_beta=args.turbulence_beta,
            perfect_fraction=args.perfect_fraction,
            elliptical_fraction=args.elliptical_fraction,
            perftextured_fraction=args.perftextured_fraction,
            elliptextured_fraction=args.elliptextured_fraction,
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
            axis_ratio=args.axis_ratio,
            axis_ratio_min=args.axis_ratio_min,
            axis_ratio_max=args.axis_ratio_max,
            compression=args.compression,
            compression_min=args.compression_min,
            compression_max=args.compression_max,
            tail_strength=args.tail_strength,
            tail_strength_min=args.tail_strength_min,
            tail_strength_max=args.tail_strength_max,
            transition=args.transition,
            transition_min=args.transition_min,
            transition_max=args.transition_max,
            turbulence_alpha=args.turbulence_alpha,
            turbulence_alpha_min=args.turbulence_alpha_min,
            turbulence_alpha_max=args.turbulence_alpha_max,
            turbulence_beta=args.turbulence_beta,
            perfect_fraction=args.perfect_fraction,
            elliptical_fraction=args.elliptical_fraction,
            perftextured_fraction=args.perftextured_fraction,
            elliptextured_fraction=args.elliptextured_fraction,
            size=args.size,
            fov=args.fov,
            seed=args.seed,
            save=args.save,
        )
