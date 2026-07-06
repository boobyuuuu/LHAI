import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


SIM_RES = 128
FOV_DEG = 6.4
PIXEL_SCALE_DEG = FOV_DEG / SIM_RES
INFERNO = LinearSegmentedColormap.from_list(
    "heavy_tail_halo_inferno",
    ["#000000", "#2D005C", "#8B1A5B", "#E85D04", "#FFE66D", "#FFFFFF"],
)
DATASET_OUTPUT = "Category_I_HeavyTail_1000_128_GT.npy"


def get_log_uniform_intensity(vmin, vmax, rng):
    return 10 ** rng.uniform(np.log10(vmin), np.log10(vmax))


def make_coordinate_grid(size=SIM_RES, fov=FOV_DEG):
    pixel_size = fov / size
    axis = np.linspace(-fov / 2, fov / 2, size, endpoint=False) + pixel_size / 2
    return np.meshgrid(axis.astype(np.float32), axis.astype(np.float32))


def get_gaussian_center(center_mu=0.0, center_sigma=0.4, max_offset=2.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    while True:
        cx = rng.normal(center_mu, center_sigma)
        cy = rng.normal(center_mu, center_sigma)
        if abs(cx) < max_offset and abs(cy) < max_offset:
            return cx, cy


def make_fractal_noise(size=SIM_RES, beta=3.2, rng=None):
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


def make_anisotropic_radius(x, y, cx, cy, axis_ratio=0.8, theta=0.0):
    dx = x - cx
    dy = y - cy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    return np.sqrt(x_rot * x_rot + (y_rot / axis_ratio) ** 2).astype(np.float32, copy=False)


def make_moffat_halo(
    x,
    y,
    cx,
    cy,
    core_radius=0.22,
    profile_beta=1.35,
    axis_ratio=0.8,
    theta=0.0,
    truncation_radius=3.2,
    truncation_width=0.45,
):
    r = make_anisotropic_radius(
        x=x,
        y=y,
        cx=cx,
        cy=cy,
        axis_ratio=axis_ratio,
        theta=theta,
    )
    profile = 1.0 / (1.0 + (r / core_radius) ** 2) ** profile_beta

    if truncation_radius > 0 and truncation_width > 0:
        z = np.clip((r - truncation_radius) / truncation_width, -60.0, 60.0)
        profile = profile / (1.0 + np.exp(z))

    return profile.astype(np.float32, copy=False), r


def apply_halo_turbulence(image, alpha=0.18, beta=3.2, rng=None):
    if alpha <= 0:
        return image.astype(np.float32, copy=False)

    noise = make_fractal_noise(size=image.shape[0], beta=beta, rng=rng)
    modulation = 1.0 - alpha + 2.0 * alpha * noise
    return (image * modulation).astype(np.float32, copy=False)


def normalize_intensity(image, intensity):
    total = image.sum(dtype=np.float64)
    if total > 0:
        image = image / total * intensity
    return image.astype(np.float32, copy=False)


def simulate_heavy_tail_halo(
    center_mu=0.0,
    center_sigma=0.4,
    max_offset=2.2,
    intensity=100.0,
    core_radius=0.22,
    profile_beta=1.35,
    axis_ratio=0.8,
    theta=None,
    truncation_radius=3.2,
    truncation_width=0.45,
    turbulence_alpha=0.18,
    turbulence_beta=3.2,
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

    if theta is None:
        theta = rng.uniform(0, np.pi)

    image, radius = make_moffat_halo(
        x=x,
        y=y,
        cx=cx,
        cy=cy,
        core_radius=core_radius,
        profile_beta=profile_beta,
        axis_ratio=axis_ratio,
        theta=theta,
        truncation_radius=truncation_radius,
        truncation_width=truncation_width,
    )
    image = apply_halo_turbulence(
        image=image,
        alpha=turbulence_alpha,
        beta=turbulence_beta,
        rng=rng,
    )
    image = normalize_intensity(image, intensity)

    params = {
        "type": "I_HeavyTail_Halo",
        "cx": cx,
        "cy": cy,
        "core_radius": core_radius,
        "profile_beta": profile_beta,
        "axis_ratio": axis_ratio,
        "theta": theta,
        "truncation_radius": truncation_radius,
        "truncation_width": truncation_width,
        "turbulence_alpha": turbulence_alpha,
        "turbulence_beta": turbulence_beta,
        "intensity": intensity,
        "radius_max": float(radius.max()),
    }
    return image, params


def simulate_random_category_i_heavy_tail(
    center_mu=0.0,
    center_sigma=0.4,
    max_offset=2.2,
    intensity_range=(10, 500),
    core_radius_range=(0.08, 0.32),
    profile_beta_range=(0.65, 2.0),
    axis_ratio_range=(0.55, 1.0),
    truncation_radius_range=(2.0, 4.8),
    truncation_width_range=(0.25, 0.75),
    turbulence_alpha_range=(0.06, 0.28),
    turbulence_beta_range=(2.6, 4.2),
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
):
    rng = np.random.default_rng(seed)
    return simulate_heavy_tail_halo(
        center_mu=center_mu,
        center_sigma=center_sigma,
        max_offset=max_offset,
        intensity=get_log_uniform_intensity(*intensity_range, rng=rng),
        core_radius=rng.uniform(*core_radius_range),
        profile_beta=rng.uniform(*profile_beta_range),
        axis_ratio=rng.uniform(*axis_ratio_range),
        theta=rng.uniform(0, np.pi),
        truncation_radius=rng.uniform(*truncation_radius_range),
        truncation_width=rng.uniform(*truncation_width_range),
        turbulence_alpha=rng.uniform(*turbulence_alpha_range),
        turbulence_beta=rng.uniform(*turbulence_beta_range),
        size=size,
        fov=fov,
        seed=rng.integers(0, np.iinfo(np.int32).max),
    )


def generate_category_i_heavy_tail_dataset(
    count=1000,
    center_mu=0.0,
    center_sigma=0.4,
    max_offset=2.2,
    intensity_range=(10, 500),
    core_radius_range=(0.08, 0.32),
    profile_beta_range=(0.65, 2.0),
    axis_ratio_range=(0.55, 1.0),
    truncation_radius_range=(2.0, 4.8),
    truncation_width_range=(0.25, 0.75),
    turbulence_alpha_range=(0.06, 0.28),
    turbulence_beta_range=(2.6, 4.2),
    size=SIM_RES,
    fov=FOV_DEG,
    output=DATASET_OUTPUT,
    seed=None,
):
    rng = np.random.default_rng(seed)
    dataset = np.empty((count, size, size), dtype=np.float32)

    for i in range(count):
        image, _ = simulate_random_category_i_heavy_tail(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity_range=intensity_range,
            core_radius_range=core_radius_range,
            profile_beta_range=profile_beta_range,
            axis_ratio_range=axis_ratio_range,
            truncation_radius_range=truncation_radius_range,
            truncation_width_range=truncation_width_range,
            turbulence_alpha_range=turbulence_alpha_range,
            turbulence_beta_range=turbulence_beta_range,
            size=size,
            fov=fov,
            seed=rng.integers(0, np.iinfo(np.int32).max),
        )
        dataset[i] = image

    np.save(output, dataset)
    return dataset


def get_flux_radius(image, fov=FOV_DEG, fraction=0.90):
    size = image.shape[0]
    x, y = make_coordinate_grid(size=size, fov=fov)
    peak_idx = np.unravel_index(np.argmax(image), image.shape)
    cx = x[peak_idx]
    cy = y[peak_idx]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).ravel()
    weights = image.astype(np.float64).ravel()
    order = np.argsort(radius)
    cumsum = np.cumsum(weights[order])
    total = cumsum[-1]
    if total <= 0:
        return 0.0
    idx = np.searchsorted(cumsum, fraction * total)
    return float(radius[order[min(idx, len(order) - 1)]])


def show_category_i_heavy_tail_source(
    random=False,
    center_mu=0.0,
    center_sigma=0.4,
    max_offset=2.2,
    intensity=100.0,
    intensity_min=10.0,
    intensity_max=500.0,
    core_radius=0.22,
    core_radius_min=0.08,
    core_radius_max=0.32,
    profile_beta=1.35,
    profile_beta_min=0.65,
    profile_beta_max=2.0,
    axis_ratio=0.8,
    axis_ratio_min=0.55,
    axis_ratio_max=1.0,
    theta=None,
    truncation_radius=3.2,
    truncation_radius_min=2.0,
    truncation_radius_max=4.8,
    truncation_width=0.45,
    truncation_width_min=0.25,
    truncation_width_max=0.75,
    turbulence_alpha=0.18,
    turbulence_alpha_min=0.06,
    turbulence_alpha_max=0.28,
    turbulence_beta=3.2,
    turbulence_beta_min=2.6,
    turbulence_beta_max=4.2,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
    save=None,
):
    if random:
        image, params = simulate_random_category_i_heavy_tail(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity_range=(intensity_min, intensity_max),
            core_radius_range=(core_radius_min, core_radius_max),
            profile_beta_range=(profile_beta_min, profile_beta_max),
            axis_ratio_range=(axis_ratio_min, axis_ratio_max),
            truncation_radius_range=(truncation_radius_min, truncation_radius_max),
            truncation_width_range=(truncation_width_min, truncation_width_max),
            turbulence_alpha_range=(turbulence_alpha_min, turbulence_alpha_max),
            turbulence_beta_range=(turbulence_beta_min, turbulence_beta_max),
            size=size,
            fov=fov,
            seed=seed,
        )
    else:
        image, params = simulate_heavy_tail_halo(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity=intensity,
            core_radius=core_radius,
            profile_beta=profile_beta,
            axis_ratio=axis_ratio,
            theta=theta,
            truncation_radius=truncation_radius,
            truncation_width=truncation_width,
            turbulence_alpha=turbulence_alpha,
            turbulence_beta=turbulence_beta,
            size=size,
            fov=fov,
            seed=seed,
        )

    total_flux = float(image.sum())
    peak_flux = float(image.max())
    r90 = get_flux_radius(image, fov=fov, fraction=0.90)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    image_plot = ax.imshow(
        image,
        origin="lower",
        cmap=INFERNO,
        extent=[-fov / 2, fov / 2, -fov / 2, fov / 2],
    )
    ax.set_title("Category I Heavy-tailed Halo Preview")
    ax.set_xlabel("RA offset (deg)")
    ax.set_ylabel("Dec offset (deg)")

    ax.text(
        0.02,
        0.98,
        f"{params['type']}\n"
        f"cx={params['cx']:.3f}°, cy={params['cy']:.3f}°\n"
        f"rc={params['core_radius']:.3f}°, beta={params['profile_beta']:.2f}, q={params['axis_ratio']:.2f}\n"
        f"theta={params['theta']:.2f}, trunc={params['truncation_radius']:.2f}°, tw={params['truncation_width']:.2f}°\n"
        f"turb={params['turbulence_alpha']:.2f}, fbm={params['turbulence_beta']:.2f}, R90={r90:.2f}°\n"
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
        plt.close(fig)
    else:
        plt.show()

    return image, params


def parse_args():
    parser = argparse.ArgumentParser(description="Preview Category-I heavy-tailed Moffat halo generation.")
    parser.add_argument("--random", action="store_true", help="Randomly sample Category-I heavy-tail physical parameters")
    parser.add_argument("--center-mu", type=float, default=0.0, help="Gaussian center distribution mean in degrees")
    parser.add_argument("--center-sigma", type=float, default=0.4, help="Gaussian center distribution sigma in degrees")
    parser.add_argument("--max-offset", type=float, default=2.2, help="Maximum absolute center offset in degrees")
    parser.add_argument("--intensity", type=float, default=100.0, help="Total source intensity for non-random preview")
    parser.add_argument("--intensity-min", type=float, default=10.0, help="Random preview minimum total intensity")
    parser.add_argument("--intensity-max", type=float, default=500.0, help="Random preview maximum total intensity")
    parser.add_argument("--core-radius", type=float, default=0.22, help="Moffat core radius in degrees")
    parser.add_argument("--core-radius-min", type=float, default=0.08, help="Random preview minimum core radius")
    parser.add_argument("--core-radius-max", type=float, default=0.32, help="Random preview maximum core radius")
    parser.add_argument("--profile-beta", type=float, default=1.35, help="Moffat power-law decay index")
    parser.add_argument("--profile-beta-min", type=float, default=0.65, help="Random preview minimum Moffat beta")
    parser.add_argument("--profile-beta-max", type=float, default=2.0, help="Random preview maximum Moffat beta")
    parser.add_argument("--axis-ratio", type=float, default=0.8, help="Minor/major halo axis ratio")
    parser.add_argument("--axis-ratio-min", type=float, default=0.55, help="Random preview minimum axis ratio")
    parser.add_argument("--axis-ratio-max", type=float, default=1.0, help="Random preview maximum axis ratio")
    parser.add_argument("--theta", type=float, default=None, help="Halo anisotropy angle in radians")
    parser.add_argument("--truncation-radius", type=float, default=3.2, help="Smooth outer truncation radius in degrees")
    parser.add_argument("--truncation-radius-min", type=float, default=2.0, help="Random preview minimum truncation radius")
    parser.add_argument("--truncation-radius-max", type=float, default=4.8, help="Random preview maximum truncation radius")
    parser.add_argument("--truncation-width", type=float, default=0.45, help="Smooth outer truncation width in degrees")
    parser.add_argument("--truncation-width-min", type=float, default=0.25, help="Random preview minimum truncation width")
    parser.add_argument("--truncation-width-max", type=float, default=0.75, help="Random preview maximum truncation width")
    parser.add_argument("--turbulence-alpha", type=float, default=0.18, help="Fractal turbulence modulation amplitude")
    parser.add_argument("--turbulence-alpha-min", type=float, default=0.06, help="Random preview minimum turbulence amplitude")
    parser.add_argument("--turbulence-alpha-max", type=float, default=0.28, help="Random preview maximum turbulence amplitude")
    parser.add_argument("--turbulence-beta", type=float, default=3.2, help="Fractal turbulence spectral index")
    parser.add_argument("--turbulence-beta-min", type=float, default=2.6, help="Random preview minimum turbulence beta")
    parser.add_argument("--turbulence-beta-max", type=float, default=4.2, help="Random preview maximum turbulence beta")
    parser.add_argument("--size", type=int, default=SIM_RES, help="Image width and height in pixels")
    parser.add_argument("--fov", type=float, default=FOV_DEG, help="Field of view in degrees")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible preview")
    parser.add_argument("--save", default=None, help="Optional path to save the plotted image")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate the Category-I heavy-tail dataset")
    parser.add_argument("--dataset-count", type=int, default=1000, help="Number of images in the dataset")
    parser.add_argument("--dataset-output", default=DATASET_OUTPUT, help="Output .npy path for the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.generate_dataset:
        data = generate_category_i_heavy_tail_dataset(
            count=args.dataset_count,
            center_mu=args.center_mu,
            center_sigma=args.center_sigma,
            max_offset=args.max_offset,
            intensity_range=(args.intensity_min, args.intensity_max),
            core_radius_range=(args.core_radius_min, args.core_radius_max),
            profile_beta_range=(args.profile_beta_min, args.profile_beta_max),
            axis_ratio_range=(args.axis_ratio_min, args.axis_ratio_max),
            truncation_radius_range=(args.truncation_radius_min, args.truncation_radius_max),
            truncation_width_range=(args.truncation_width_min, args.truncation_width_max),
            turbulence_alpha_range=(args.turbulence_alpha_min, args.turbulence_alpha_max),
            turbulence_beta_range=(args.turbulence_beta_min, args.turbulence_beta_max),
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
        show_category_i_heavy_tail_source(
            random=args.random,
            center_mu=args.center_mu,
            center_sigma=args.center_sigma,
            max_offset=args.max_offset,
            intensity=args.intensity,
            intensity_min=args.intensity_min,
            intensity_max=args.intensity_max,
            core_radius=args.core_radius,
            core_radius_min=args.core_radius_min,
            core_radius_max=args.core_radius_max,
            profile_beta=args.profile_beta,
            profile_beta_min=args.profile_beta_min,
            profile_beta_max=args.profile_beta_max,
            axis_ratio=args.axis_ratio,
            axis_ratio_min=args.axis_ratio_min,
            axis_ratio_max=args.axis_ratio_max,
            theta=args.theta,
            truncation_radius=args.truncation_radius,
            truncation_radius_min=args.truncation_radius_min,
            truncation_radius_max=args.truncation_radius_max,
            truncation_width=args.truncation_width,
            truncation_width_min=args.truncation_width_min,
            truncation_width_max=args.truncation_width_max,
            turbulence_alpha=args.turbulence_alpha,
            turbulence_alpha_min=args.turbulence_alpha_min,
            turbulence_alpha_max=args.turbulence_alpha_max,
            turbulence_beta=args.turbulence_beta,
            turbulence_beta_min=args.turbulence_beta_min,
            turbulence_beta_max=args.turbulence_beta_max,
            size=args.size,
            fov=args.fov,
            seed=args.seed,
            save=args.save,
        )
