import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


SIM_RES = 128
FOV_DEG = 6.4
PIXEL_SCALE_DEG = FOV_DEG / SIM_RES
INFERNO = LinearSegmentedColormap.from_list(
    "fragmented_disk_inferno",
    ["#000000", "#2D005C", "#8B1A5B", "#E85D04", "#FFE66D", "#FFFFFF"],
)
DATASET_OUTPUT = "Type_D_FragmentedDisk_1000_128_GT.npy"


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


def make_base_disk(x, y, cx, cy, radius):
    disk = ((x - cx) ** 2 + (y - cy) ** 2 <= radius * radius).astype(np.float32)
    return disk


def make_gaussian_kernel(sigma_pixels):
    if sigma_pixels <= 0:
        return np.array([1.0], dtype=np.float32)

    radius = max(1, int(np.ceil(4 * sigma_pixels)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (coords / sigma_pixels) ** 2)
    kernel = kernel / kernel.sum(dtype=np.float64)
    return kernel.astype(np.float32, copy=False)


def apply_gaussian_blur(image, sigma_deg, pixel_scale_deg=PIXEL_SCALE_DEG):
    sigma_pixels = sigma_deg / pixel_scale_deg
    kernel = make_gaussian_kernel(sigma_pixels)
    radius = len(kernel) // 2
    if radius == 0:
        return image.astype(np.float32, copy=True)

    padded = np.pad(image.astype(np.float32, copy=False), ((0, 0), (radius, radius)), mode="constant")
    tmp = np.empty_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        tmp[i] = np.convolve(padded[i], kernel, mode="valid")

    padded = np.pad(tmp, ((radius, radius), (0, 0)), mode="constant")
    blurred = np.empty_like(image, dtype=np.float32)
    for j in range(image.shape[1]):
        blurred[:, j] = np.convolve(padded[:, j], kernel, mode="valid")

    return blurred.astype(np.float32, copy=False)


def make_fractal_noise(size=SIM_RES, beta=2.0, rng=None):
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


def apply_fractal_cavities(image, threshold=0.45, cavity_floor=0.05, beta=2.0, power=1.0, rng=None):
    noise = make_fractal_noise(size=image.shape[0], beta=beta, rng=rng)
    dense = np.clip((noise - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    if power != 1.0:
        dense = dense ** power
    modulation = cavity_floor + (1.0 - cavity_floor) * dense
    return image * modulation, noise


def apply_asymmetric_gradient(x, y, image, cx, cy, phi=np.pi / 4, strength=0.35, floor=0.25):
    dx = x - cx
    dy = y - cy
    s = dx * np.cos(phi) + dy * np.sin(phi)
    norm = np.max(np.abs(s))
    if norm > 0:
        s = s / norm
    gradient = np.clip(1.0 + strength * s, floor, None)
    return image * gradient


def normalize_intensity(image, intensity):
    total = image.sum(dtype=np.float64)
    if total > 0:
        image = image / total * intensity
    return image.astype(np.float32, copy=False)


def simulate_fragmented_disk(
    center_mu=0.0,
    center_sigma=0.4,
    max_offset=2.4,
    intensity=200.0,
    radius=0.6,
    blur_sigma=0.10,
    cavity_threshold=0.45,
    cavity_floor=0.05,
    cavity_beta=2.0,
    cavity_power=1.0,
    gradient_phi=np.pi / 4,
    gradient_strength=0.35,
    gradient_floor=0.25,
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

    image = make_base_disk(x=x, y=y, cx=cx, cy=cy, radius=radius)
    image = apply_gaussian_blur(image=image, sigma_deg=blur_sigma, pixel_scale_deg=fov / size)
    image, noise = apply_fractal_cavities(
        image=image,
        threshold=cavity_threshold,
        cavity_floor=cavity_floor,
        beta=cavity_beta,
        power=cavity_power,
        rng=rng,
    )
    image = apply_asymmetric_gradient(
        x=x,
        y=y,
        image=image,
        cx=cx,
        cy=cy,
        phi=gradient_phi,
        strength=gradient_strength,
        floor=gradient_floor,
    )
    image = normalize_intensity(image, intensity)

    params = {
        "type": "D_FragmentedDisk",
        "cx": cx,
        "cy": cy,
        "radius": radius,
        "blur_sigma": blur_sigma,
        "cavity_threshold": cavity_threshold,
        "cavity_floor": cavity_floor,
        "cavity_beta": cavity_beta,
        "cavity_power": cavity_power,
        "gradient_phi": gradient_phi,
        "gradient_strength": gradient_strength,
        "intensity": intensity,
        "noise_mean": float(noise.mean()),
    }
    return image, params


def simulate_random_type_d(
    center_mu=0.0,
    center_sigma=0.4,
    max_offset=2.4,
    intensity_range=(20, 800),
    radius_range=(0.4, 0.8),
    blur_sigma_range=(0.05, 0.15),
    cavity_threshold_range=(0.35, 0.60),
    cavity_floor_range=(0.0, 0.12),
    cavity_beta_range=(1.6, 2.6),
    cavity_power_range=(0.8, 1.6),
    gradient_strength_range=(0.15, 0.55),
    gradient_floor=0.25,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
):
    rng = np.random.default_rng(seed)
    return simulate_fragmented_disk(
        center_mu=center_mu,
        center_sigma=center_sigma,
        max_offset=max_offset,
        intensity=get_log_uniform_intensity(*intensity_range, rng=rng),
        radius=rng.uniform(*radius_range),
        blur_sigma=rng.uniform(*blur_sigma_range),
        cavity_threshold=rng.uniform(*cavity_threshold_range),
        cavity_floor=rng.uniform(*cavity_floor_range),
        cavity_beta=rng.uniform(*cavity_beta_range),
        cavity_power=rng.uniform(*cavity_power_range),
        gradient_phi=rng.uniform(0, 2 * np.pi),
        gradient_strength=rng.uniform(*gradient_strength_range),
        gradient_floor=gradient_floor,
        size=size,
        fov=fov,
        seed=rng.integers(0, np.iinfo(np.int32).max),
    )


def generate_type_d_disk_dataset(
    count=1000,
    center_mu=0.0,
    center_sigma=0.4,
    max_offset=2.4,
    intensity_range=(20, 800),
    radius_range=(0.4, 0.8),
    blur_sigma_range=(0.05, 0.15),
    cavity_threshold_range=(0.35, 0.60),
    cavity_floor_range=(0.0, 0.12),
    cavity_beta_range=(1.6, 2.6),
    cavity_power_range=(0.8, 1.6),
    gradient_strength_range=(0.15, 0.55),
    gradient_floor=0.25,
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
        image = make_base_disk(
            x=x,
            y=y,
            cx=cx,
            cy=cy,
            radius=rng.uniform(*radius_range),
        )
        image = apply_gaussian_blur(
            image=image,
            sigma_deg=rng.uniform(*blur_sigma_range),
            pixel_scale_deg=fov / size,
        )
        image, _ = apply_fractal_cavities(
            image=image,
            threshold=rng.uniform(*cavity_threshold_range),
            cavity_floor=rng.uniform(*cavity_floor_range),
            beta=rng.uniform(*cavity_beta_range),
            power=rng.uniform(*cavity_power_range),
            rng=rng,
        )
        image = apply_asymmetric_gradient(
            x=x,
            y=y,
            image=image,
            cx=cx,
            cy=cy,
            phi=rng.uniform(0, 2 * np.pi),
            strength=rng.uniform(*gradient_strength_range),
            floor=gradient_floor,
        )
        dataset[i] = normalize_intensity(image, get_log_uniform_intensity(*intensity_range, rng=rng))

    np.save(output, dataset)
    return dataset


def show_type_d_source(
    random=False,
    center_mu=0.0,
    center_sigma=0.4,
    max_offset=2.4,
    intensity=200.0,
    intensity_min=20.0,
    intensity_max=800.0,
    radius=0.6,
    radius_min=0.4,
    radius_max=0.8,
    blur_sigma=0.10,
    blur_sigma_min=0.05,
    blur_sigma_max=0.15,
    cavity_threshold=0.45,
    cavity_threshold_min=0.35,
    cavity_threshold_max=0.60,
    cavity_floor=0.05,
    cavity_floor_min=0.0,
    cavity_floor_max=0.12,
    cavity_beta=2.0,
    cavity_beta_min=1.6,
    cavity_beta_max=2.6,
    cavity_power=1.0,
    cavity_power_min=0.8,
    cavity_power_max=1.6,
    gradient_phi=np.pi / 4,
    gradient_strength=0.35,
    gradient_strength_min=0.15,
    gradient_strength_max=0.55,
    gradient_floor=0.25,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
    save=None,
):
    if random:
        image, params = simulate_random_type_d(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity_range=(intensity_min, intensity_max),
            radius_range=(radius_min, radius_max),
            blur_sigma_range=(blur_sigma_min, blur_sigma_max),
            cavity_threshold_range=(cavity_threshold_min, cavity_threshold_max),
            cavity_floor_range=(cavity_floor_min, cavity_floor_max),
            cavity_beta_range=(cavity_beta_min, cavity_beta_max),
            cavity_power_range=(cavity_power_min, cavity_power_max),
            gradient_strength_range=(gradient_strength_min, gradient_strength_max),
            gradient_floor=gradient_floor,
            size=size,
            fov=fov,
            seed=seed,
        )
    else:
        image, params = simulate_fragmented_disk(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity=intensity,
            radius=radius,
            blur_sigma=blur_sigma,
            cavity_threshold=cavity_threshold,
            cavity_floor=cavity_floor,
            cavity_beta=cavity_beta,
            cavity_power=cavity_power,
            gradient_phi=gradient_phi,
            gradient_strength=gradient_strength,
            gradient_floor=gradient_floor,
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
    ax.set_title("Type D Fragmented Disk Preview")
    ax.set_xlabel("RA offset (deg)")
    ax.set_ylabel("Dec offset (deg)")

    ax.text(
        0.02,
        0.98,
        f"{params['type']}\n"
        f"cx={params['cx']:.3f}°, cy={params['cy']:.3f}°\n"
        f"R={params['radius']:.3f}°, blur={params['blur_sigma']:.3f}°\n"
        f"thr={params['cavity_threshold']:.2f}, floor={params['cavity_floor']:.2f}, beta={params['cavity_beta']:.2f}\n"
        f"grad={params['gradient_strength']:.2f}, phi={params['gradient_phi']:.2f}\n"
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
    parser = argparse.ArgumentParser(description="Preview Type-D fragmented disk source generation.")
    parser.add_argument("--random", action="store_true", help="Randomly sample Type-D physical parameters")
    parser.add_argument("--center-mu", type=float, default=0.0, help="Gaussian center distribution mean in degrees")
    parser.add_argument("--center-sigma", type=float, default=0.8, help="Gaussian center distribution sigma in degrees")
    parser.add_argument("--max-offset", type=float, default=2.4, help="Maximum absolute center offset in degrees")
    parser.add_argument("--intensity", type=float, default=200.0, help="Total source intensity for non-random preview")
    parser.add_argument("--intensity-min", type=float, default=20.0, help="Random preview minimum total intensity")
    parser.add_argument("--intensity-max", type=float, default=800.0, help="Random preview maximum total intensity")
    parser.add_argument("--radius", type=float, default=0.4, help="Disk radius in degrees")
    parser.add_argument("--radius-min", type=float, default=0.2, help="Random preview minimum disk radius in degrees")
    parser.add_argument("--radius-max", type=float, default=0.8, help="Random preview maximum disk radius in degrees")
    parser.add_argument("--blur-sigma", type=float, default=0.10, help="Edge-softening Gaussian blur sigma in degrees")
    parser.add_argument("--blur-sigma-min", type=float, default=0.05, help="Random preview minimum blur sigma in degrees")
    parser.add_argument("--blur-sigma-max", type=float, default=0.15, help="Random preview maximum blur sigma in degrees")
    parser.add_argument("--cavity-threshold", type=float, default=0.45, help="Noise threshold below which cavities are carved")
    parser.add_argument("--cavity-threshold-min", type=float, default=0.35, help="Random preview minimum cavity threshold")
    parser.add_argument("--cavity-threshold-max", type=float, default=0.60, help="Random preview maximum cavity threshold")
    parser.add_argument("--cavity-floor", type=float, default=0.05, help="Residual flux fraction inside cavities")
    parser.add_argument("--cavity-floor-min", type=float, default=0.0, help="Random preview minimum cavity floor")
    parser.add_argument("--cavity-floor-max", type=float, default=0.12, help="Random preview maximum cavity floor")
    parser.add_argument("--cavity-beta", type=float, default=2.0, help="Fractal noise spectral index for cavities")
    parser.add_argument("--cavity-beta-min", type=float, default=1.6, help="Random preview minimum cavity beta")
    parser.add_argument("--cavity-beta-max", type=float, default=2.6, help="Random preview maximum cavity beta")
    parser.add_argument("--cavity-power", type=float, default=1.0, help="Dense clump contrast power after thresholding")
    parser.add_argument("--cavity-power-min", type=float, default=0.8, help="Random preview minimum cavity power")
    parser.add_argument("--cavity-power-max", type=float, default=1.6, help="Random preview maximum cavity power")
    parser.add_argument("--gradient-phi", type=float, default=np.pi / 4, help="Bright-side gradient direction angle in radians")
    parser.add_argument("--gradient-strength", type=float, default=0.35, help="Large-scale linear gradient strength")
    parser.add_argument("--gradient-strength-min", type=float, default=0.15, help="Random preview minimum gradient strength")
    parser.add_argument("--gradient-strength-max", type=float, default=0.55, help="Random preview maximum gradient strength")
    parser.add_argument("--gradient-floor", type=float, default=0.25, help="Minimum multiplicative gradient value")
    parser.add_argument("--size", type=int, default=SIM_RES, help="Image width and height in pixels")
    parser.add_argument("--fov", type=float, default=FOV_DEG, help="Field of view in degrees")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible preview")
    parser.add_argument("--save", default=None, help="Optional path to save the plotted image")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate the Type-D fragmented disk dataset")
    parser.add_argument("--dataset-count", type=int, default=1000, help="Number of images in the dataset")
    parser.add_argument("--dataset-output", default=DATASET_OUTPUT, help="Output .npy path for the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.generate_dataset:
        data = generate_type_d_disk_dataset(
            count=args.dataset_count,
            center_mu=args.center_mu,
            center_sigma=args.center_sigma,
            max_offset=args.max_offset,
            intensity_range=(args.intensity_min, args.intensity_max),
            radius_range=(args.radius_min, args.radius_max),
            blur_sigma_range=(args.blur_sigma_min, args.blur_sigma_max),
            cavity_threshold_range=(args.cavity_threshold_min, args.cavity_threshold_max),
            cavity_floor_range=(args.cavity_floor_min, args.cavity_floor_max),
            cavity_beta_range=(args.cavity_beta_min, args.cavity_beta_max),
            cavity_power_range=(args.cavity_power_min, args.cavity_power_max),
            gradient_strength_range=(args.gradient_strength_min, args.gradient_strength_max),
            gradient_floor=args.gradient_floor,
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
        show_type_d_source(
            random=args.random,
            center_mu=args.center_mu,
            center_sigma=args.center_sigma,
            max_offset=args.max_offset,
            intensity=args.intensity,
            intensity_min=args.intensity_min,
            intensity_max=args.intensity_max,
            radius=args.radius,
            radius_min=args.radius_min,
            radius_max=args.radius_max,
            blur_sigma=args.blur_sigma,
            blur_sigma_min=args.blur_sigma_min,
            blur_sigma_max=args.blur_sigma_max,
            cavity_threshold=args.cavity_threshold,
            cavity_threshold_min=args.cavity_threshold_min,
            cavity_threshold_max=args.cavity_threshold_max,
            cavity_floor=args.cavity_floor,
            cavity_floor_min=args.cavity_floor_min,
            cavity_floor_max=args.cavity_floor_max,
            cavity_beta=args.cavity_beta,
            cavity_beta_min=args.cavity_beta_min,
            cavity_beta_max=args.cavity_beta_max,
            cavity_power=args.cavity_power,
            cavity_power_min=args.cavity_power_min,
            cavity_power_max=args.cavity_power_max,
            gradient_phi=args.gradient_phi,
            gradient_strength=args.gradient_strength,
            gradient_strength_min=args.gradient_strength_min,
            gradient_strength_max=args.gradient_strength_max,
            gradient_floor=args.gradient_floor,
            size=args.size,
            fov=args.fov,
            seed=args.seed,
            save=args.save,
        )
