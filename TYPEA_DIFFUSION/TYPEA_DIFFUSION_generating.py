import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


SIM_RES = 128
FOV_DEG = 6.4
PIXEL_SCALE_DEG = FOV_DEG / SIM_RES
INFERNO = LinearSegmentedColormap.from_list(
    "diffusion_halo_inferno",
    ["#000000", "#2D005C", "#8B1A5B", "#E85D04", "#FFE66D", "#FFFFFF"],
)
DATASET_OUTPUT = "Type_A_DIFFUSION_1000_128_GT.npy"


def get_log_uniform_intensity(vmin, vmax, rng):
    return 10 ** rng.uniform(np.log10(vmin), np.log10(vmax))


def make_coordinate_grid(size=SIM_RES, fov=FOV_DEG):
    pixel_size = fov / size
    axis = np.linspace(-fov / 2, fov / 2, size, endpoint=False) + pixel_size / 2
    return np.meshgrid(axis.astype(np.float32), axis.astype(np.float32))


def get_gaussian_center(center_mu=0.0, center_sigma=0.4, max_offset=2.4, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    while True:
        cx = rng.normal(center_mu, center_sigma)
        cy = rng.normal(center_mu, center_sigma)
        if abs(cx) < max_offset and abs(cy) < max_offset:
            return cx, cy


def make_fractal_noise(size=SIM_RES, beta=3.8, rng=None):
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


def make_anisotropic_diffusion_radius(x, y, cx, cy, lambda1, lambda2, theta):
    dx = x - cx
    dy = y - cy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    return np.sqrt((x_rot / lambda1) ** 2 + (y_rot / lambda2) ** 2).astype(np.float32, copy=False)


def make_powerlaw_diffusion_profile(
    x,
    y,
    cx,
    cy,
    lambda1=0.6,
    lambda2=0.5,
    theta=0.0,
    core_radius=0.38,
    profile_beta=2.2,
    cutoff_radius=1.0,
    cutoff_width=0.08,
):
    r_diff = make_anisotropic_diffusion_radius(
        x=x,
        y=y,
        cx=cx,
        cy=cy,
        lambda1=lambda1,
        lambda2=lambda2,
        theta=theta,
    )
    profile = 1.0 / (1.0 + (r_diff / core_radius) ** profile_beta)

    if cutoff_radius > 0 and cutoff_width > 0:
        z = np.clip((r_diff - cutoff_radius) / cutoff_width, -60.0, 60.0)
        profile = profile / (1.0 + np.exp(z))

    return profile.astype(np.float32, copy=False), r_diff


def make_fbm_background(size=SIM_RES, beta=3.8, floor=0.20, contrast=1.0, rng=None):
    noise = make_fractal_noise(size=size, beta=beta, rng=rng)
    background = floor + (1.0 - floor) * noise
    if contrast != 1.0:
        mean = background.mean(dtype=np.float64)
        background = mean + contrast * (background - mean)
        background = np.clip(background, 0.0, None)
    return background.astype(np.float32, copy=False)


def normalize_intensity(image, intensity):
    total = image.sum(dtype=np.float64)
    if total > 0:
        image = image / total * intensity
    return image.astype(np.float32, copy=False)


def combine_source_and_background(source, background, intensity, background_fraction=0.10):
    background_fraction = np.clip(background_fraction, 0.0, 0.95)
    source = normalize_intensity(source, intensity * (1.0 - background_fraction))
    background = normalize_intensity(background, intensity * background_fraction)
    return (source + background).astype(np.float32, copy=False)


def simulate_diffusion_halo(
    center_mu=0.0,
    center_sigma=0.4,
    max_offset=2.4,
    intensity=100.0,
    lambda1=0.6,
    lambda2=None,
    axis_ratio=0.85,
    theta=None,
    core_radius=0.38,
    profile_beta=2.2,
    cutoff_radius=1.0,
    cutoff_width=0.08,
    background_fraction=0.10,
    background_beta=3.8,
    background_floor=0.20,
    background_contrast=1.0,
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

    if lambda2 is None:
        lambda2 = lambda1 * axis_ratio
    if theta is None:
        theta = rng.uniform(0, np.pi)

    source, r_diff = make_powerlaw_diffusion_profile(
        x=x,
        y=y,
        cx=cx,
        cy=cy,
        lambda1=lambda1,
        lambda2=lambda2,
        theta=theta,
        core_radius=core_radius,
        profile_beta=profile_beta,
        cutoff_radius=cutoff_radius,
        cutoff_width=cutoff_width,
    )
    background = make_fbm_background(
        size=size,
        beta=background_beta,
        floor=background_floor,
        contrast=background_contrast,
        rng=rng,
    )
    image = combine_source_and_background(
        source=source,
        background=background,
        intensity=intensity,
        background_fraction=background_fraction,
    )

    params = {
        "type": "A_DIFFUSION",
        "cx": cx,
        "cy": cy,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "axis_ratio": lambda2 / lambda1,
        "theta": theta,
        "core_radius": core_radius,
        "profile_beta": profile_beta,
        "cutoff_radius": cutoff_radius,
        "cutoff_width": cutoff_width,
        "background_fraction": background_fraction,
        "background_beta": background_beta,
        "background_floor": background_floor,
        "background_contrast": background_contrast,
        "intensity": intensity,
        "r_diff_max": float(r_diff.max()),
    }
    return image, params


def simulate_random_type_a_diffusion(
    center_mu=0.0,
    center_sigma=0.4,
    max_offset=2.4,
    intensity_range=(10, 500),
    lambda1_range=(0.3, 0.8),
    axis_ratio_range=(0.75, 1.0),
    core_radius_range=(0.25, 0.55),
    profile_beta_range=(1.5, 3.0),
    cutoff_radius_range=(0.85, 1.15),
    cutoff_width_range=(0.04, 0.14),
    background_fraction_range=(0.04, 0.18),
    background_beta_range=(3.2, 4.4),
    background_floor_range=(0.10, 0.35),
    background_contrast_range=(0.6, 1.4),
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
):
    rng = np.random.default_rng(seed)
    lambda1 = rng.uniform(*lambda1_range)
    return simulate_diffusion_halo(
        center_mu=center_mu,
        center_sigma=center_sigma,
        max_offset=max_offset,
        intensity=get_log_uniform_intensity(*intensity_range, rng=rng),
        lambda1=lambda1,
        lambda2=lambda1 * rng.uniform(*axis_ratio_range),
        theta=rng.uniform(0, np.pi),
        core_radius=rng.uniform(*core_radius_range),
        profile_beta=rng.uniform(*profile_beta_range),
        cutoff_radius=rng.uniform(*cutoff_radius_range),
        cutoff_width=rng.uniform(*cutoff_width_range),
        background_fraction=rng.uniform(*background_fraction_range),
        background_beta=rng.uniform(*background_beta_range),
        background_floor=rng.uniform(*background_floor_range),
        background_contrast=rng.uniform(*background_contrast_range),
        size=size,
        fov=fov,
        seed=rng.integers(0, np.iinfo(np.int32).max),
    )


def generate_type_a_diffusion_dataset(
    count=1000,
    center_mu=0.0,
    center_sigma=0.4,
    max_offset=2.4,
    intensity_range=(10, 500),
    lambda1_range=(0.3, 0.8),
    axis_ratio_range=(0.75, 1.0),
    core_radius_range=(0.25, 0.55),
    profile_beta_range=(1.5, 3.0),
    cutoff_radius_range=(0.85, 1.15),
    cutoff_width_range=(0.04, 0.14),
    background_fraction_range=(0.04, 0.18),
    background_beta_range=(3.2, 4.4),
    background_floor_range=(0.10, 0.35),
    background_contrast_range=(0.6, 1.4),
    size=SIM_RES,
    fov=FOV_DEG,
    output=DATASET_OUTPUT,
    seed=None,
):
    rng = np.random.default_rng(seed)
    dataset = np.empty((count, size, size), dtype=np.float32)

    for i in range(count):
        image, _ = simulate_random_type_a_diffusion(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity_range=intensity_range,
            lambda1_range=lambda1_range,
            axis_ratio_range=axis_ratio_range,
            core_radius_range=core_radius_range,
            profile_beta_range=profile_beta_range,
            cutoff_radius_range=cutoff_radius_range,
            cutoff_width_range=cutoff_width_range,
            background_fraction_range=background_fraction_range,
            background_beta_range=background_beta_range,
            background_floor_range=background_floor_range,
            background_contrast_range=background_contrast_range,
            size=size,
            fov=fov,
            seed=rng.integers(0, np.iinfo(np.int32).max),
        )
        dataset[i] = image

    np.save(output, dataset)
    return dataset


def get_peak_visible_radius(image, fov=FOV_DEG, peak_fraction=0.02):
    size = image.shape[0]
    x, y = make_coordinate_grid(size=size, fov=fov)
    peak_idx = np.unravel_index(np.argmax(image), image.shape)
    cx = x[peak_idx]
    cy = y[peak_idx]
    mask = image >= image.max() * peak_fraction
    if not np.any(mask):
        return 0.0
    return float(np.sqrt((x[mask] - cx) ** 2 + (y[mask] - cy) ** 2).max())


def show_type_a_diffusion_source(
    random=False,
    center_mu=0.0,
    center_sigma=0.4,
    max_offset=2.4,
    intensity=100.0,
    intensity_min=10.0,
    intensity_max=500.0,
    lambda1=0.6,
    lambda1_min=0.3,
    lambda1_max=0.8,
    lambda2=None,
    axis_ratio=0.85,
    axis_ratio_min=0.75,
    axis_ratio_max=1.0,
    theta=None,
    core_radius=0.38,
    core_radius_min=0.25,
    core_radius_max=0.55,
    profile_beta=2.2,
    profile_beta_min=1.5,
    profile_beta_max=3.0,
    cutoff_radius=1.0,
    cutoff_radius_min=0.85,
    cutoff_radius_max=1.15,
    cutoff_width=0.08,
    cutoff_width_min=0.04,
    cutoff_width_max=0.14,
    background_fraction=0.10,
    background_fraction_min=0.04,
    background_fraction_max=0.18,
    background_beta=3.8,
    background_beta_min=3.2,
    background_beta_max=4.4,
    background_floor=0.20,
    background_floor_min=0.10,
    background_floor_max=0.35,
    background_contrast=1.0,
    background_contrast_min=0.6,
    background_contrast_max=1.4,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
    save=None,
):
    if random:
        image, params = simulate_random_type_a_diffusion(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity_range=(intensity_min, intensity_max),
            lambda1_range=(lambda1_min, lambda1_max),
            axis_ratio_range=(axis_ratio_min, axis_ratio_max),
            core_radius_range=(core_radius_min, core_radius_max),
            profile_beta_range=(profile_beta_min, profile_beta_max),
            cutoff_radius_range=(cutoff_radius_min, cutoff_radius_max),
            cutoff_width_range=(cutoff_width_min, cutoff_width_max),
            background_fraction_range=(background_fraction_min, background_fraction_max),
            background_beta_range=(background_beta_min, background_beta_max),
            background_floor_range=(background_floor_min, background_floor_max),
            background_contrast_range=(background_contrast_min, background_contrast_max),
            size=size,
            fov=fov,
            seed=seed,
        )
    else:
        image, params = simulate_diffusion_halo(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity=intensity,
            lambda1=lambda1,
            lambda2=lambda2,
            axis_ratio=axis_ratio,
            theta=theta,
            core_radius=core_radius,
            profile_beta=profile_beta,
            cutoff_radius=cutoff_radius,
            cutoff_width=cutoff_width,
            background_fraction=background_fraction,
            background_beta=background_beta,
            background_floor=background_floor,
            background_contrast=background_contrast,
            size=size,
            fov=fov,
            seed=seed,
        )

    total_flux = float(image.sum())
    peak_flux = float(image.max())
    visible_radius = get_peak_visible_radius(image, fov=fov, peak_fraction=0.02)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    image_plot = ax.imshow(
        image,
        origin="lower",
        cmap=INFERNO,
        extent=[-fov / 2, fov / 2, -fov / 2, fov / 2],
    )
    ax.set_title("Type A Diffusion Halo Preview")
    ax.set_xlabel("RA offset (deg)")
    ax.set_ylabel("Dec offset (deg)")

    ax.text(
        0.02,
        0.98,
        f"{params['type']}\n"
        f"cx={params['cx']:.3f}°, cy={params['cy']:.3f}°\n"
        f"lambda1={params['lambda1']:.2f}°, lambda2={params['lambda2']:.2f}°, q={params['axis_ratio']:.2f}\n"
        f"rc={params['core_radius']:.2f}, beta={params['profile_beta']:.2f}, theta={params['theta']:.2f}\n"
        f"cut={params['cutoff_radius']:.2f}, bg={params['background_fraction']:.2f}, fbm={params['background_beta']:.2f}\n"
        f"R2%={visible_radius:.2f}°, Flux={total_flux:.3f}, Peak={peak_flux:.3f}",
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
        plt.close(fig)
    else:
        plt.show()

    return image, params


def parse_args():
    parser = argparse.ArgumentParser(description="Preview Type-A diffusion halo generation.")
    parser.add_argument("--random", action="store_true", help="Randomly sample Type-A diffusion physical parameters")
    parser.add_argument("--center-mu", type=float, default=0.0, help="Gaussian center distribution mean in degrees")
    parser.add_argument("--center-sigma", type=float, default=0.4, help="Gaussian center distribution sigma in degrees")
    parser.add_argument("--max-offset", type=float, default=2.4, help="Maximum absolute center offset in degrees")
    parser.add_argument("--intensity", type=float, default=100.0, help="Total source plus background intensity for non-random preview")
    parser.add_argument("--intensity-min", type=float, default=10.0, help="Random preview minimum total intensity")
    parser.add_argument("--intensity-max", type=float, default=500.0, help="Random preview maximum total intensity")
    parser.add_argument("--lambda1", type=float, default=0.6, help="Major diffusion-axis visible scale in degrees")
    parser.add_argument("--lambda1-min", type=float, default=0.3, help="Random preview minimum major diffusion-axis scale")
    parser.add_argument("--lambda1-max", type=float, default=0.8, help="Random preview maximum major diffusion-axis scale")
    parser.add_argument("--lambda2", type=float, default=None, help="Minor diffusion-axis scale in degrees; defaults to lambda1 * axis_ratio")
    parser.add_argument("--axis-ratio", type=float, default=0.85, help="Minor/major diffusion-axis scale ratio")
    parser.add_argument("--axis-ratio-min", type=float, default=0.75, help="Random preview minimum minor/major ratio")
    parser.add_argument("--axis-ratio-max", type=float, default=1.0, help="Random preview maximum minor/major ratio")
    parser.add_argument("--theta", type=float, default=None, help="Diffusion tensor rotation angle in radians")
    parser.add_argument("--core-radius", type=float, default=0.38, help="Power-law core radius in diffusion-coordinate units")
    parser.add_argument("--core-radius-min", type=float, default=0.25, help="Random preview minimum core radius")
    parser.add_argument("--core-radius-max", type=float, default=0.55, help="Random preview maximum core radius")
    parser.add_argument("--profile-beta", type=float, default=2.2, help="Power-law decay steepness")
    parser.add_argument("--profile-beta-min", type=float, default=1.5, help="Random preview minimum power-law beta")
    parser.add_argument("--profile-beta-max", type=float, default=3.0, help="Random preview maximum power-law beta")
    parser.add_argument("--cutoff-radius", type=float, default=1.0, help="Smooth truncation radius in diffusion-coordinate units")
    parser.add_argument("--cutoff-radius-min", type=float, default=0.85, help="Random preview minimum cutoff radius")
    parser.add_argument("--cutoff-radius-max", type=float, default=1.15, help="Random preview maximum cutoff radius")
    parser.add_argument("--cutoff-width", type=float, default=0.08, help="Smooth truncation transition width")
    parser.add_argument("--cutoff-width-min", type=float, default=0.04, help="Random preview minimum cutoff width")
    parser.add_argument("--cutoff-width-max", type=float, default=0.14, help="Random preview maximum cutoff width")
    parser.add_argument("--background-fraction", type=float, default=0.10, help="Fraction of total flux assigned to fBm ISM background")
    parser.add_argument("--background-fraction-min", type=float, default=0.04, help="Random preview minimum background flux fraction")
    parser.add_argument("--background-fraction-max", type=float, default=0.18, help="Random preview maximum background flux fraction")
    parser.add_argument("--background-beta", type=float, default=3.8, help="fBm background spectral index")
    parser.add_argument("--background-beta-min", type=float, default=3.2, help="Random preview minimum fBm beta")
    parser.add_argument("--background-beta-max", type=float, default=4.4, help="Random preview maximum fBm beta")
    parser.add_argument("--background-floor", type=float, default=0.20, help="Smooth floor mixed into fBm background before normalization")
    parser.add_argument("--background-floor-min", type=float, default=0.10, help="Random preview minimum background floor")
    parser.add_argument("--background-floor-max", type=float, default=0.35, help="Random preview maximum background floor")
    parser.add_argument("--background-contrast", type=float, default=1.0, help="fBm background contrast multiplier")
    parser.add_argument("--background-contrast-min", type=float, default=0.6, help="Random preview minimum background contrast")
    parser.add_argument("--background-contrast-max", type=float, default=1.4, help="Random preview maximum background contrast")
    parser.add_argument("--size", type=int, default=SIM_RES, help="Image width and height in pixels")
    parser.add_argument("--fov", type=float, default=FOV_DEG, help="Field of view in degrees")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible preview")
    parser.add_argument("--save", default=None, help="Optional path to save the plotted image")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate the Type-A diffusion dataset")
    parser.add_argument("--dataset-count", type=int, default=1000, help="Number of images in the dataset")
    parser.add_argument("--dataset-output", default=DATASET_OUTPUT, help="Output .npy path for the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.generate_dataset:
        data = generate_type_a_diffusion_dataset(
            count=args.dataset_count,
            center_mu=args.center_mu,
            center_sigma=args.center_sigma,
            max_offset=args.max_offset,
            intensity_range=(args.intensity_min, args.intensity_max),
            lambda1_range=(args.lambda1_min, args.lambda1_max),
            axis_ratio_range=(args.axis_ratio_min, args.axis_ratio_max),
            core_radius_range=(args.core_radius_min, args.core_radius_max),
            profile_beta_range=(args.profile_beta_min, args.profile_beta_max),
            cutoff_radius_range=(args.cutoff_radius_min, args.cutoff_radius_max),
            cutoff_width_range=(args.cutoff_width_min, args.cutoff_width_max),
            background_fraction_range=(args.background_fraction_min, args.background_fraction_max),
            background_beta_range=(args.background_beta_min, args.background_beta_max),
            background_floor_range=(args.background_floor_min, args.background_floor_max),
            background_contrast_range=(args.background_contrast_min, args.background_contrast_max),
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
        show_type_a_diffusion_source(
            random=args.random,
            center_mu=args.center_mu,
            center_sigma=args.center_sigma,
            max_offset=args.max_offset,
            intensity=args.intensity,
            intensity_min=args.intensity_min,
            intensity_max=args.intensity_max,
            lambda1=args.lambda1,
            lambda1_min=args.lambda1_min,
            lambda1_max=args.lambda1_max,
            lambda2=args.lambda2,
            axis_ratio=args.axis_ratio,
            axis_ratio_min=args.axis_ratio_min,
            axis_ratio_max=args.axis_ratio_max,
            theta=args.theta,
            core_radius=args.core_radius,
            core_radius_min=args.core_radius_min,
            core_radius_max=args.core_radius_max,
            profile_beta=args.profile_beta,
            profile_beta_min=args.profile_beta_min,
            profile_beta_max=args.profile_beta_max,
            cutoff_radius=args.cutoff_radius,
            cutoff_radius_min=args.cutoff_radius_min,
            cutoff_radius_max=args.cutoff_radius_max,
            cutoff_width=args.cutoff_width,
            cutoff_width_min=args.cutoff_width_min,
            cutoff_width_max=args.cutoff_width_max,
            background_fraction=args.background_fraction,
            background_fraction_min=args.background_fraction_min,
            background_fraction_max=args.background_fraction_max,
            background_beta=args.background_beta,
            background_beta_min=args.background_beta_min,
            background_beta_max=args.background_beta_max,
            background_floor=args.background_floor,
            background_floor_min=args.background_floor_min,
            background_floor_max=args.background_floor_max,
            background_contrast=args.background_contrast,
            background_contrast_min=args.background_contrast_min,
            background_contrast_max=args.background_contrast_max,
            size=args.size,
            fov=args.fov,
            seed=args.seed,
            save=args.save,
        )
