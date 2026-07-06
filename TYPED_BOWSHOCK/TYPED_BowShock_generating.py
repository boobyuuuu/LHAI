import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


SIM_RES = 128
FOV_DEG = 6.4
PIXEL_SCALE_DEG = FOV_DEG / SIM_RES
INFERNO = LinearSegmentedColormap.from_list(
    "bowshock_tail_inferno",
    ["#000000", "#2D005C", "#8B1A5B", "#E85D04", "#FFE66D", "#FFFFFF"],
)
DATASET_OUTPUT = "Type_A_BowShock_1000_128_GT.npy"


def get_log_uniform_intensity(vmin, vmax, rng):
    return 10 ** rng.uniform(np.log10(vmin), np.log10(vmax))


def make_coordinate_grid(size=SIM_RES, fov=FOV_DEG):
    pixel_size = fov / size
    axis = np.linspace(-fov / 2, fov / 2, size, endpoint=False) + pixel_size / 2
    return np.meshgrid(axis.astype(np.float32), axis.astype(np.float32))


def get_gaussian_center(center_mu=0.0, center_sigma=0.45, max_offset=2.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    while True:
        cx = rng.normal(center_mu, center_sigma)
        cy = rng.normal(center_mu, center_sigma)
        if abs(cx) < max_offset and abs(cy) < max_offset:
            return cx, cy


def make_fractal_noise(size=SIM_RES, beta=2.7, rng=None):
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


def rotate_to_shock_frame(x, y, cx, cy, theta):
    dx = x - cx
    dy = y - cy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    return x_rot, y_rot


def make_piecewise_bowshock_profile(
    x_rot,
    y_rot,
    sigma=0.28,
    transverse_sigma=0.18,
    compression_factor=0.38,
    elongation_factor=3.2,
    shell_radius=0.25,
    shell_width=0.055,
    tail_power=1.35,
    tail_floor=0.08,
):
    sigma_front = sigma * compression_factor
    sigma_back = sigma * elongation_factor
    sigma_x = np.where(x_rot >= 0.0, sigma_front, sigma_back)

    core = np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / transverse_sigma) ** 2))

    front_gate = 1.0 / (1.0 + np.exp(-x_rot / max(shell_width, 1e-6)))
    bow_center = shell_radius - 0.55 * (y_rot / max(transverse_sigma, 1e-6)) ** 2 * sigma_front
    shell = np.exp(-0.5 * ((x_rot - bow_center) / shell_width) ** 2)
    shell *= np.exp(-0.5 * (y_rot / (1.45 * transverse_sigma)) ** 2) * front_gate

    tail_s = np.maximum(-x_rot, 0.0)
    tail_width = transverse_sigma * (1.0 + 0.9 * tail_s / max(sigma_back, 1e-6))
    tail = np.exp(-0.5 * (y_rot / tail_width) ** 2) / (1.0 + (tail_s / max(sigma, 1e-6)) ** tail_power)
    tail *= 1.0 / (1.0 + np.exp(x_rot / max(sigma_front, 1e-6)))

    image = 0.45 * core + 0.40 * shell + 0.35 * tail
    image = np.maximum(image - tail_floor * np.exp(-0.5 * ((x_rot / sigma_front) ** 2 + (y_rot / transverse_sigma) ** 2)), 0.0)
    return image.astype(np.float32, copy=False)


def apply_plasma_turbulence(image, alpha=0.30, beta=2.7, rng=None):
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


def simulate_bowshock_tail(
    center_mu=0.0,
    center_sigma=0.45,
    max_offset=2.2,
    intensity=100.0,
    sigma=0.28,
    transverse_sigma=0.18,
    shock_theta=None,
    compression_factor=0.38,
    elongation_factor=3.2,
    shell_radius=0.25,
    shell_width=0.055,
    tail_power=1.35,
    turbulence_alpha=0.30,
    turbulence_beta=2.7,
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

    if shock_theta is None:
        shock_theta = rng.uniform(0, 2 * np.pi)

    x_rot, y_rot = rotate_to_shock_frame(x=x, y=y, cx=cx, cy=cy, theta=shock_theta)
    image = make_piecewise_bowshock_profile(
        x_rot=x_rot,
        y_rot=y_rot,
        sigma=sigma,
        transverse_sigma=transverse_sigma,
        compression_factor=compression_factor,
        elongation_factor=elongation_factor,
        shell_radius=shell_radius,
        shell_width=shell_width,
        tail_power=tail_power,
    )
    image = apply_plasma_turbulence(
        image=image,
        alpha=turbulence_alpha,
        beta=turbulence_beta,
        rng=rng,
    )
    image = normalize_intensity(image, intensity)

    params = {
        "type": "A_BowShock",
        "cx": cx,
        "cy": cy,
        "sigma": sigma,
        "transverse_sigma": transverse_sigma,
        "shock_theta": shock_theta,
        "compression_factor": compression_factor,
        "elongation_factor": elongation_factor,
        "shell_radius": shell_radius,
        "shell_width": shell_width,
        "tail_power": tail_power,
        "turbulence_alpha": turbulence_alpha,
        "turbulence_beta": turbulence_beta,
        "intensity": intensity,
    }
    return image, params


def simulate_random_type_a_bowshock(
    center_mu=0.0,
    center_sigma=0.45,
    max_offset=2.2,
    intensity_range=(10, 500),
    sigma_range=(0.16, 0.38),
    transverse_sigma_range=(0.10, 0.26),
    compression_factor_range=(0.22, 0.58),
    elongation_factor_range=(2.2, 5.0),
    shell_radius_range=(0.14, 0.38),
    shell_width_range=(0.035, 0.090),
    tail_power_range=(1.0, 1.9),
    turbulence_alpha_range=(0.12, 0.42),
    turbulence_beta_range=(2.1, 3.5),
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
):
    rng = np.random.default_rng(seed)
    return simulate_bowshock_tail(
        center_mu=center_mu,
        center_sigma=center_sigma,
        max_offset=max_offset,
        intensity=get_log_uniform_intensity(*intensity_range, rng=rng),
        sigma=rng.uniform(*sigma_range),
        transverse_sigma=rng.uniform(*transverse_sigma_range),
        shock_theta=rng.uniform(0, 2 * np.pi),
        compression_factor=rng.uniform(*compression_factor_range),
        elongation_factor=rng.uniform(*elongation_factor_range),
        shell_radius=rng.uniform(*shell_radius_range),
        shell_width=rng.uniform(*shell_width_range),
        tail_power=rng.uniform(*tail_power_range),
        turbulence_alpha=rng.uniform(*turbulence_alpha_range),
        turbulence_beta=rng.uniform(*turbulence_beta_range),
        size=size,
        fov=fov,
        seed=rng.integers(0, np.iinfo(np.int32).max),
    )


def generate_type_a_bowshock_dataset(
    count=1000,
    center_mu=0.0,
    center_sigma=0.45,
    max_offset=2.2,
    intensity_range=(10, 500),
    sigma_range=(0.16, 0.38),
    transverse_sigma_range=(0.10, 0.26),
    compression_factor_range=(0.22, 0.58),
    elongation_factor_range=(2.2, 5.0),
    shell_radius_range=(0.14, 0.38),
    shell_width_range=(0.035, 0.090),
    tail_power_range=(1.0, 1.9),
    turbulence_alpha_range=(0.12, 0.42),
    turbulence_beta_range=(2.1, 3.5),
    size=SIM_RES,
    fov=FOV_DEG,
    output=DATASET_OUTPUT,
    seed=None,
):
    rng = np.random.default_rng(seed)
    dataset = np.empty((count, size, size), dtype=np.float32)

    for i in range(count):
        image, _ = simulate_random_type_a_bowshock(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity_range=intensity_range,
            sigma_range=sigma_range,
            transverse_sigma_range=transverse_sigma_range,
            compression_factor_range=compression_factor_range,
            elongation_factor_range=elongation_factor_range,
            shell_radius_range=shell_radius_range,
            shell_width_range=shell_width_range,
            tail_power_range=tail_power_range,
            turbulence_alpha_range=turbulence_alpha_range,
            turbulence_beta_range=turbulence_beta_range,
            size=size,
            fov=fov,
            seed=rng.integers(0, np.iinfo(np.int32).max),
        )
        dataset[i] = image

    np.save(output, dataset)
    return dataset


def show_type_a_bowshock_source(
    random=False,
    center_mu=0.0,
    center_sigma=0.45,
    max_offset=2.2,
    intensity=100.0,
    intensity_min=10.0,
    intensity_max=500.0,
    sigma=0.28,
    sigma_min=0.16,
    sigma_max=0.38,
    transverse_sigma=0.18,
    transverse_sigma_min=0.10,
    transverse_sigma_max=0.26,
    shock_theta=None,
    compression_factor=0.38,
    compression_factor_min=0.22,
    compression_factor_max=0.58,
    elongation_factor=3.2,
    elongation_factor_min=2.2,
    elongation_factor_max=5.0,
    shell_radius=0.25,
    shell_radius_min=0.14,
    shell_radius_max=0.38,
    shell_width=0.055,
    shell_width_min=0.035,
    shell_width_max=0.090,
    tail_power=1.35,
    tail_power_min=1.0,
    tail_power_max=1.9,
    turbulence_alpha=0.30,
    turbulence_alpha_min=0.12,
    turbulence_alpha_max=0.42,
    turbulence_beta=2.7,
    turbulence_beta_min=2.1,
    turbulence_beta_max=3.5,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
    save=None,
):
    if random:
        image, params = simulate_random_type_a_bowshock(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity_range=(intensity_min, intensity_max),
            sigma_range=(sigma_min, sigma_max),
            transverse_sigma_range=(transverse_sigma_min, transverse_sigma_max),
            compression_factor_range=(compression_factor_min, compression_factor_max),
            elongation_factor_range=(elongation_factor_min, elongation_factor_max),
            shell_radius_range=(shell_radius_min, shell_radius_max),
            shell_width_range=(shell_width_min, shell_width_max),
            tail_power_range=(tail_power_min, tail_power_max),
            turbulence_alpha_range=(turbulence_alpha_min, turbulence_alpha_max),
            turbulence_beta_range=(turbulence_beta_min, turbulence_beta_max),
            size=size,
            fov=fov,
            seed=seed,
        )
    else:
        image, params = simulate_bowshock_tail(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity=intensity,
            sigma=sigma,
            transverse_sigma=transverse_sigma,
            shock_theta=shock_theta,
            compression_factor=compression_factor,
            elongation_factor=elongation_factor,
            shell_radius=shell_radius,
            shell_width=shell_width,
            tail_power=tail_power,
            turbulence_alpha=turbulence_alpha,
            turbulence_beta=turbulence_beta,
            size=size,
            fov=fov,
            seed=seed,
        )

    total_flux = float(image.sum())
    peak_flux = float(image.max())
    tail_length = params["sigma"] * params["elongation_factor"]
    front_scale = params["sigma"] * params["compression_factor"]

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    image_plot = ax.imshow(
        image,
        origin="lower",
        cmap=INFERNO,
        extent=[-fov / 2, fov / 2, -fov / 2, fov / 2],
    )
    ax.set_title("Type A BowShock Cometary-tail Preview")
    ax.set_xlabel("RA offset (deg)")
    ax.set_ylabel("Dec offset (deg)")

    ax.text(
        0.02,
        0.98,
        f"{params['type']}\n"
        f"cx={params['cx']:.3f}°, cy={params['cy']:.3f}°\n"
        f"theta={params['shock_theta']:.2f}, sigma={params['sigma']:.2f}°, trans={params['transverse_sigma']:.2f}°\n"
        f"front={front_scale:.2f}°, tail={tail_length:.2f}°, comp={params['compression_factor']:.2f}, elong={params['elongation_factor']:.2f}\n"
        f"shell={params['shell_radius']:.2f}°, width={params['shell_width']:.3f}°, turb={params['turbulence_alpha']:.2f}\n"
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
    parser = argparse.ArgumentParser(description="Preview Type-A piecewise BowShock cometary-tail generation.")
    parser.add_argument("--random", action="store_true", help="Randomly sample Type-A BowShock physical parameters")
    parser.add_argument("--center-mu", type=float, default=0.0, help="Gaussian center distribution mean in degrees")
    parser.add_argument("--center-sigma", type=float, default=0.45, help="Gaussian center distribution sigma in degrees")
    parser.add_argument("--max-offset", type=float, default=2.2, help="Maximum absolute center offset in degrees")
    parser.add_argument("--intensity", type=float, default=100.0, help="Total source intensity for non-random preview")
    parser.add_argument("--intensity-min", type=float, default=10.0, help="Random preview minimum total intensity")
    parser.add_argument("--intensity-max", type=float, default=500.0, help="Random preview maximum total intensity")
    parser.add_argument("--sigma", type=float, default=0.28, help="Base axial sigma in degrees")
    parser.add_argument("--sigma-min", type=float, default=0.16, help="Random preview minimum base axial sigma")
    parser.add_argument("--sigma-max", type=float, default=0.38, help="Random preview maximum base axial sigma")
    parser.add_argument("--transverse-sigma", type=float, default=0.18, help="Cross-axis sigma in degrees")
    parser.add_argument("--transverse-sigma-min", type=float, default=0.10, help="Random preview minimum cross-axis sigma")
    parser.add_argument("--transverse-sigma-max", type=float, default=0.26, help="Random preview maximum cross-axis sigma")
    parser.add_argument("--shock-theta", type=float, default=None, help="Motion/shock axis angle in radians")
    parser.add_argument("--compression-factor", type=float, default=0.38, help="Forward compression factor")
    parser.add_argument("--compression-factor-min", type=float, default=0.22, help="Random preview minimum forward compression factor")
    parser.add_argument("--compression-factor-max", type=float, default=0.58, help="Random preview maximum forward compression factor")
    parser.add_argument("--elongation-factor", type=float, default=3.2, help="Backward tail elongation factor")
    parser.add_argument("--elongation-factor-min", type=float, default=2.2, help="Random preview minimum backward elongation factor")
    parser.add_argument("--elongation-factor-max", type=float, default=5.0, help="Random preview maximum backward elongation factor")
    parser.add_argument("--shell-radius", type=float, default=0.25, help="Bow-shock shell stand-off radius in degrees")
    parser.add_argument("--shell-radius-min", type=float, default=0.14, help="Random preview minimum shell stand-off radius")
    parser.add_argument("--shell-radius-max", type=float, default=0.38, help="Random preview maximum shell stand-off radius")
    parser.add_argument("--shell-width", type=float, default=0.055, help="Bow-shock shell thickness in degrees")
    parser.add_argument("--shell-width-min", type=float, default=0.035, help="Random preview minimum shell thickness")
    parser.add_argument("--shell-width-max", type=float, default=0.090, help="Random preview maximum shell thickness")
    parser.add_argument("--tail-power", type=float, default=1.35, help="Power-law tail fading exponent")
    parser.add_argument("--tail-power-min", type=float, default=1.0, help="Random preview minimum tail fading exponent")
    parser.add_argument("--tail-power-max", type=float, default=1.9, help="Random preview maximum tail fading exponent")
    parser.add_argument("--turbulence-alpha", type=float, default=0.30, help="Plasma turbulence modulation amplitude")
    parser.add_argument("--turbulence-alpha-min", type=float, default=0.12, help="Random preview minimum turbulence amplitude")
    parser.add_argument("--turbulence-alpha-max", type=float, default=0.42, help="Random preview maximum turbulence amplitude")
    parser.add_argument("--turbulence-beta", type=float, default=2.7, help="Plasma turbulence fractal spectral index")
    parser.add_argument("--turbulence-beta-min", type=float, default=2.1, help="Random preview minimum turbulence beta")
    parser.add_argument("--turbulence-beta-max", type=float, default=3.5, help="Random preview maximum turbulence beta")
    parser.add_argument("--size", type=int, default=SIM_RES, help="Image width and height in pixels")
    parser.add_argument("--fov", type=float, default=FOV_DEG, help="Field of view in degrees")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible preview")
    parser.add_argument("--save", default=None, help="Optional path to save the plotted image")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate the Type-A BowShock dataset")
    parser.add_argument("--dataset-count", type=int, default=1000, help="Number of images in the dataset")
    parser.add_argument("--dataset-output", default=DATASET_OUTPUT, help="Output .npy path for the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.generate_dataset:
        data = generate_type_a_bowshock_dataset(
            count=args.dataset_count,
            center_mu=args.center_mu,
            center_sigma=args.center_sigma,
            max_offset=args.max_offset,
            intensity_range=(args.intensity_min, args.intensity_max),
            sigma_range=(args.sigma_min, args.sigma_max),
            transverse_sigma_range=(args.transverse_sigma_min, args.transverse_sigma_max),
            compression_factor_range=(args.compression_factor_min, args.compression_factor_max),
            elongation_factor_range=(args.elongation_factor_min, args.elongation_factor_max),
            shell_radius_range=(args.shell_radius_min, args.shell_radius_max),
            shell_width_range=(args.shell_width_min, args.shell_width_max),
            tail_power_range=(args.tail_power_min, args.tail_power_max),
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
        show_type_a_bowshock_source(
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
            transverse_sigma=args.transverse_sigma,
            transverse_sigma_min=args.transverse_sigma_min,
            transverse_sigma_max=args.transverse_sigma_max,
            shock_theta=args.shock_theta,
            compression_factor=args.compression_factor,
            compression_factor_min=args.compression_factor_min,
            compression_factor_max=args.compression_factor_max,
            elongation_factor=args.elongation_factor,
            elongation_factor_min=args.elongation_factor_min,
            elongation_factor_max=args.elongation_factor_max,
            shell_radius=args.shell_radius,
            shell_radius_min=args.shell_radius_min,
            shell_radius_max=args.shell_radius_max,
            shell_width=args.shell_width,
            shell_width_min=args.shell_width_min,
            shell_width_max=args.shell_width_max,
            tail_power=args.tail_power,
            tail_power_min=args.tail_power_min,
            tail_power_max=args.tail_power_max,
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
