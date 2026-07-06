import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm


SIM_RES = 128
FOV_DEG = 6.4
PIXEL_SCALE_DEG = FOV_DEG / SIM_RES
INFERNO = LinearSegmentedColormap.from_list(
    "type_c_mc_inferno",
    ["#000000", "#101030", "#2D005C", "#7A1F5C", "#E85D04", "#FFE66D", "#FFFFFF"],
)
DATASET_OUTPUT = "Type_C_MC_1000_128_GT.npy"


def get_log_uniform_intensity(vmin, vmax, rng):
    return 10 ** rng.uniform(np.log10(vmin), np.log10(vmax))


def make_coordinate_grid(size=SIM_RES, fov=FOV_DEG):
    pixel_size = fov / size
    axis = np.linspace(-fov / 2, fov / 2, size, endpoint=False) + pixel_size / 2
    return np.meshgrid(axis.astype(np.float32), axis.astype(np.float32))


def standardize_field(field):
    field = field.astype(np.float32, copy=False)
    mean = field.mean(dtype=np.float64)
    std = field.std(dtype=np.float64)
    if std <= 0:
        return np.zeros_like(field, dtype=np.float32)
    return ((field - mean) / std).astype(np.float32, copy=False)


def robust_rescale(field, low_percentile=0.5, high_percentile=99.8):
    lo = float(np.percentile(field, low_percentile))
    hi = float(np.percentile(field, high_percentile))
    if hi <= lo:
        hi = float(field.max())
    if hi <= lo:
        return np.zeros_like(field, dtype=np.float32)
    scaled = np.clip((field - lo) / (hi - lo), 0.0, 1.0)
    return scaled.astype(np.float32, copy=False)


def make_fbm_noise(size=SIM_RES, beta=2.8, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    white = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    spectrum = np.fft.fft2(white)

    freq = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freq, freq)
    k = np.sqrt(fx * fx + fy * fy)
    k[0, 0] = np.inf
    filt = k ** (-0.5 * beta)
    filt[0, 0] = 0.0

    noise = np.real(np.fft.ifft2(spectrum * filt))
    return standardize_field(noise)


def make_lognormal_clump_field(size=SIM_RES, beta=2.8, alpha=1.45, clip_percentile=99.8, rng=None):
    noise = make_fbm_noise(size=size, beta=beta, rng=rng)
    density = np.exp(alpha * np.clip(noise, -4.0, 4.0))
    density = robust_rescale(density, low_percentile=0.2, high_percentile=clip_percentile)
    return density, noise


def make_ridged_multifractal(
    size=SIM_RES,
    beta=2.4,
    ridge_width=1.15,
    ridge_power=2.7,
    octaves=3,
    persistence=0.55,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    ridges = np.zeros((size, size), dtype=np.float32)
    weight_sum = 0.0
    for octave in range(octaves):
        octave_beta = max(1.2, beta - 0.25 * octave)
        noise = make_fbm_noise(size=size, beta=octave_beta, rng=rng)
        ridge = 1.0 - np.abs(np.clip(noise / ridge_width, -1.0, 1.0))
        ridge = np.clip(ridge, 0.0, 1.0) ** ridge_power
        weight = persistence ** octave
        ridges += weight * ridge.astype(np.float32, copy=False)
        weight_sum += weight

    if weight_sum > 0:
        ridges /= weight_sum
    return robust_rescale(ridges, low_percentile=5.0, high_percentile=99.8)


def rotate_to_frame(x, y, cx, cy, theta):
    dx = x - cx
    dy = y - cy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    u = dx * cos_t + dy * sin_t
    v = -dx * sin_t + dy * cos_t
    return u, v


def make_superbubble_void_mask(
    x,
    y,
    count=3,
    sigma_range=(0.45, 1.10),
    depth_range=(0.55, 0.90),
    axis_ratio_range=(0.65, 1.0),
    void_floor=0.05,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    mask = np.ones_like(x, dtype=np.float32)
    voids = []
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    for _ in range(count):
        cx = rng.uniform(xmin, xmax)
        cy = rng.uniform(ymin, ymax)
        sigma = rng.uniform(*sigma_range)
        depth = rng.uniform(*depth_range)
        axis_ratio = rng.uniform(*axis_ratio_range)
        theta = rng.uniform(0, np.pi)
        u, v = rotate_to_frame(x=x, y=y, cx=cx, cy=cy, theta=theta)
        bowl = np.exp(-0.5 * ((u / sigma) ** 2 + (v / (sigma * axis_ratio)) ** 2))
        mask *= (1.0 - depth * bowl).astype(np.float32, copy=False)
        voids.append(
            {
                "cx": float(cx),
                "cy": float(cy),
                "sigma": float(sigma),
                "depth": float(depth),
                "axis_ratio": float(axis_ratio),
                "theta": float(theta),
            }
        )

    mask = np.clip(mask, void_floor, 1.0)
    return mask.astype(np.float32, copy=False), voids


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

    padded = np.pad(image.astype(np.float32, copy=False), ((0, 0), (radius, radius)), mode="reflect")
    tmp = np.empty_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        tmp[i] = np.convolve(padded[i], kernel, mode="valid")

    padded = np.pad(tmp, ((radius, radius), (0, 0)), mode="reflect")
    blurred = np.empty_like(image, dtype=np.float32)
    for j in range(image.shape[1]):
        blurred[:, j] = np.convolve(padded[:, j], kernel, mode="valid")

    return blurred.astype(np.float32, copy=False)


def normalize_intensity(image, intensity):
    total = image.sum(dtype=np.float64)
    if total > 0:
        image = image / total * intensity
    return image.astype(np.float32, copy=False)


def simulate_molecular_cloud_cr_sea(
    intensity=300.0,
    clump_beta=2.8,
    clump_alpha=1.45,
    filament_beta=2.4,
    filament_strength=1.15,
    filament_power=2.7,
    filament_octaves=3,
    ridge_width=1.15,
    background_floor=0.018,
    void_count=3,
    void_sigma_range=(0.45, 1.10),
    void_depth_range=(0.55, 0.90),
    void_floor=0.05,
    diffusion_sigma=0.075,
    clip_percentile=99.8,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
):
    rng = np.random.default_rng(seed)
    x, y = make_coordinate_grid(size=size, fov=fov)

    clumps, _ = make_lognormal_clump_field(
        size=size,
        beta=clump_beta,
        alpha=clump_alpha,
        clip_percentile=clip_percentile,
        rng=rng,
    )
    filaments = make_ridged_multifractal(
        size=size,
        beta=filament_beta,
        ridge_width=ridge_width,
        ridge_power=filament_power,
        octaves=filament_octaves,
        rng=rng,
    )
    void_mask, voids = make_superbubble_void_mask(
        x=x,
        y=y,
        count=void_count,
        sigma_range=void_sigma_range,
        depth_range=void_depth_range,
        void_floor=void_floor,
        rng=rng,
    )

    gas = background_floor + clumps * (1.0 + filament_strength * filaments)
    gas = gas * void_mask
    gas = apply_gaussian_blur(gas, sigma_deg=diffusion_sigma, pixel_scale_deg=fov / size)
    gas = np.clip(gas, 0.0, None)
    image = normalize_intensity(gas, intensity)

    params = {
        "type": "C_MC",
        "intensity": intensity,
        "clump_beta": clump_beta,
        "clump_alpha": clump_alpha,
        "filament_beta": filament_beta,
        "filament_strength": filament_strength,
        "filament_power": filament_power,
        "filament_octaves": filament_octaves,
        "ridge_width": ridge_width,
        "background_floor": background_floor,
        "void_count": void_count,
        "void_sigma_range": void_sigma_range,
        "void_depth_range": void_depth_range,
        "void_floor": void_floor,
        "diffusion_sigma": diffusion_sigma,
        "diffusion_sigma_pixels": diffusion_sigma / (fov / size),
        "clip_percentile": clip_percentile,
        "voids": voids,
    }
    return image, params


def simulate_random_type_c_mc(
    intensity_range=(120, 900),
    clump_beta_range=(2.5, 3.2),
    clump_alpha_range=(1.05, 1.95),
    filament_beta_range=(2.0, 2.9),
    filament_strength_range=(0.65, 1.80),
    filament_power_range=(2.0, 3.8),
    filament_octaves=3,
    ridge_width_range=(0.95, 1.35),
    background_floor_range=(0.006, 0.040),
    void_count_range=(2, 5),
    void_sigma_range=(0.35, 1.25),
    void_depth_range=(0.45, 0.94),
    void_floor_range=(0.02, 0.10),
    diffusion_sigma_range=(0.05, 0.10),
    clip_percentile_range=(99.5, 99.9),
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
):
    rng = np.random.default_rng(seed)
    void_count = int(rng.integers(void_count_range[0], void_count_range[1] + 1))
    return simulate_molecular_cloud_cr_sea(
        intensity=get_log_uniform_intensity(*intensity_range, rng=rng),
        clump_beta=rng.uniform(*clump_beta_range),
        clump_alpha=rng.uniform(*clump_alpha_range),
        filament_beta=rng.uniform(*filament_beta_range),
        filament_strength=rng.uniform(*filament_strength_range),
        filament_power=rng.uniform(*filament_power_range),
        filament_octaves=filament_octaves,
        ridge_width=rng.uniform(*ridge_width_range),
        background_floor=rng.uniform(*background_floor_range),
        void_count=void_count,
        void_sigma_range=void_sigma_range,
        void_depth_range=void_depth_range,
        void_floor=rng.uniform(*void_floor_range),
        diffusion_sigma=rng.uniform(*diffusion_sigma_range),
        clip_percentile=rng.uniform(*clip_percentile_range),
        size=size,
        fov=fov,
        seed=rng.integers(0, np.iinfo(np.int32).max),
    )


def generate_type_c_mc_dataset(
    count=1000,
    intensity_range=(120, 900),
    clump_beta_range=(2.5, 3.2),
    clump_alpha_range=(1.05, 1.95),
    filament_beta_range=(2.0, 2.9),
    filament_strength_range=(0.65, 1.80),
    filament_power_range=(2.0, 3.8),
    filament_octaves=3,
    ridge_width_range=(0.95, 1.35),
    background_floor_range=(0.006, 0.040),
    void_count_range=(2, 5),
    void_sigma_range=(0.35, 1.25),
    void_depth_range=(0.45, 0.94),
    void_floor_range=(0.02, 0.10),
    diffusion_sigma_range=(0.05, 0.10),
    clip_percentile_range=(99.5, 99.9),
    size=SIM_RES,
    fov=FOV_DEG,
    output=DATASET_OUTPUT,
    seed=None,
):
    rng = np.random.default_rng(seed)
    dataset = np.empty((count, size, size), dtype=np.float32)

    for i in range(count):
        image, _ = simulate_random_type_c_mc(
            intensity_range=intensity_range,
            clump_beta_range=clump_beta_range,
            clump_alpha_range=clump_alpha_range,
            filament_beta_range=filament_beta_range,
            filament_strength_range=filament_strength_range,
            filament_power_range=filament_power_range,
            filament_octaves=filament_octaves,
            ridge_width_range=ridge_width_range,
            background_floor_range=background_floor_range,
            void_count_range=void_count_range,
            void_sigma_range=void_sigma_range,
            void_depth_range=void_depth_range,
            void_floor_range=void_floor_range,
            diffusion_sigma_range=diffusion_sigma_range,
            clip_percentile_range=clip_percentile_range,
            size=size,
            fov=fov,
            seed=rng.integers(0, np.iinfo(np.int32).max),
        )
        dataset[i] = image

    np.save(output, dataset)
    return dataset


def show_type_c_mc_source(
    random=False,
    intensity=300.0,
    intensity_min=120.0,
    intensity_max=900.0,
    clump_beta=2.8,
    clump_beta_min=2.5,
    clump_beta_max=3.2,
    clump_alpha=1.45,
    clump_alpha_min=1.05,
    clump_alpha_max=1.95,
    filament_beta=2.4,
    filament_beta_min=2.0,
    filament_beta_max=2.9,
    filament_strength=1.15,
    filament_strength_min=0.65,
    filament_strength_max=1.80,
    filament_power=2.7,
    filament_power_min=2.0,
    filament_power_max=3.8,
    filament_octaves=3,
    ridge_width=1.15,
    ridge_width_min=0.95,
    ridge_width_max=1.35,
    background_floor=0.018,
    background_floor_min=0.006,
    background_floor_max=0.040,
    void_count=3,
    void_count_min=2,
    void_count_max=5,
    void_sigma_min=0.45,
    void_sigma_max=1.10,
    void_depth_min=0.55,
    void_depth_max=0.90,
    void_floor=0.05,
    void_floor_min=0.02,
    void_floor_max=0.10,
    diffusion_sigma=0.075,
    diffusion_sigma_min=0.05,
    diffusion_sigma_max=0.10,
    clip_percentile=99.8,
    clip_percentile_min=99.5,
    clip_percentile_max=99.9,
    display_gamma=0.62,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
    save=None,
):
    if random:
        image, params = simulate_random_type_c_mc(
            intensity_range=(intensity_min, intensity_max),
            clump_beta_range=(clump_beta_min, clump_beta_max),
            clump_alpha_range=(clump_alpha_min, clump_alpha_max),
            filament_beta_range=(filament_beta_min, filament_beta_max),
            filament_strength_range=(filament_strength_min, filament_strength_max),
            filament_power_range=(filament_power_min, filament_power_max),
            filament_octaves=filament_octaves,
            ridge_width_range=(ridge_width_min, ridge_width_max),
            background_floor_range=(background_floor_min, background_floor_max),
            void_count_range=(void_count_min, void_count_max),
            void_sigma_range=(void_sigma_min, void_sigma_max),
            void_depth_range=(void_depth_min, void_depth_max),
            void_floor_range=(void_floor_min, void_floor_max),
            diffusion_sigma_range=(diffusion_sigma_min, diffusion_sigma_max),
            clip_percentile_range=(clip_percentile_min, clip_percentile_max),
            size=size,
            fov=fov,
            seed=seed,
        )
    else:
        image, params = simulate_molecular_cloud_cr_sea(
            intensity=intensity,
            clump_beta=clump_beta,
            clump_alpha=clump_alpha,
            filament_beta=filament_beta,
            filament_strength=filament_strength,
            filament_power=filament_power,
            filament_octaves=filament_octaves,
            ridge_width=ridge_width,
            background_floor=background_floor,
            void_count=void_count,
            void_sigma_range=(void_sigma_min, void_sigma_max),
            void_depth_range=(void_depth_min, void_depth_max),
            void_floor=void_floor,
            diffusion_sigma=diffusion_sigma,
            clip_percentile=clip_percentile,
            size=size,
            fov=fov,
            seed=seed,
        )

    total_flux = float(image.sum())
    peak_flux = float(image.max())
    vmax = float(np.percentile(image, 99.75))
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    image_plot = ax.imshow(
        image,
        origin="lower",
        cmap=INFERNO,
        norm=PowerNorm(gamma=display_gamma, vmin=0.0, vmax=vmax),
        extent=[-fov / 2, fov / 2, -fov / 2, fov / 2],
    )
    ax.set_title("Type C MC: Molecular Cloud + CR Sea Preview")
    ax.set_xlabel("RA offset (deg)")
    ax.set_ylabel("Dec offset (deg)")

    ax.text(
        0.02,
        0.98,
        f"{params['type']}\n"
        f"clump: beta={params['clump_beta']:.2f}, alpha={params['clump_alpha']:.2f}\n"
        f"ridge: beta={params['filament_beta']:.2f}, strength={params['filament_strength']:.2f}, power={params['filament_power']:.2f}\n"
        f"voids={params['void_count']}, floor={params['void_floor']:.2f}\n"
        f"CR cutoff sigma={params['diffusion_sigma']:.3f}° ({params['diffusion_sigma_pixels']:.1f}px)\n"
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
    parser = argparse.ArgumentParser(description="Preview Type-C molecular cloud and cosmic-ray sea generation.")
    parser.add_argument("--random", action="store_true", help="Randomly sample Type-C MC physical parameters")
    parser.add_argument("--intensity", type=float, default=300.0, help="Total intensity for non-random preview")
    parser.add_argument("--intensity-min", type=float, default=120.0, help="Random preview minimum total intensity")
    parser.add_argument("--intensity-max", type=float, default=900.0, help="Random preview maximum total intensity")
    parser.add_argument("--clump-beta", type=float, default=2.8, help="Log-normal fBm clump power-spectrum index")
    parser.add_argument("--clump-beta-min", type=float, default=2.5, help="Random preview minimum clump beta")
    parser.add_argument("--clump-beta-max", type=float, default=3.2, help="Random preview maximum clump beta")
    parser.add_argument("--clump-alpha", type=float, default=1.45, help="Log-normal density contrast / Mach-number proxy")
    parser.add_argument("--clump-alpha-min", type=float, default=1.05, help="Random preview minimum clump alpha")
    parser.add_argument("--clump-alpha-max", type=float, default=1.95, help="Random preview maximum clump alpha")
    parser.add_argument("--filament-beta", type=float, default=2.4, help="Ridged fBm filament spectral index")
    parser.add_argument("--filament-beta-min", type=float, default=2.0, help="Random preview minimum filament beta")
    parser.add_argument("--filament-beta-max", type=float, default=2.9, help="Random preview maximum filament beta")
    parser.add_argument("--filament-strength", type=float, default=1.15, help="Filament multiplicative brightness strength")
    parser.add_argument("--filament-strength-min", type=float, default=0.65, help="Random preview minimum filament strength")
    parser.add_argument("--filament-strength-max", type=float, default=1.80, help="Random preview maximum filament strength")
    parser.add_argument("--filament-power", type=float, default=2.7, help="Ridge sharpening power")
    parser.add_argument("--filament-power-min", type=float, default=2.0, help="Random preview minimum ridge power")
    parser.add_argument("--filament-power-max", type=float, default=3.8, help="Random preview maximum ridge power")
    parser.add_argument("--filament-octaves", type=int, default=3, help="Number of ridged fBm octaves")
    parser.add_argument("--ridge-width", type=float, default=1.15, help="Width around fBm zero-crossings retained as ridges")
    parser.add_argument("--ridge-width-min", type=float, default=0.95, help="Random preview minimum ridge width")
    parser.add_argument("--ridge-width-max", type=float, default=1.35, help="Random preview maximum ridge width")
    parser.add_argument("--background-floor", type=float, default=0.018, help="Residual CR-sea/gas floor before void carving")
    parser.add_argument("--background-floor-min", type=float, default=0.006, help="Random preview minimum background floor")
    parser.add_argument("--background-floor-max", type=float, default=0.040, help="Random preview maximum background floor")
    parser.add_argument("--void-count", type=int, default=3, help="Number of broad superbubble voids for non-random preview")
    parser.add_argument("--void-count-min", type=int, default=2, help="Random preview minimum void count")
    parser.add_argument("--void-count-max", type=int, default=5, help="Random preview maximum void count")
    parser.add_argument("--void-sigma-min", type=float, default=0.45, help="Minimum superbubble Gaussian sigma in degrees")
    parser.add_argument("--void-sigma-max", type=float, default=1.10, help="Maximum superbubble Gaussian sigma in degrees")
    parser.add_argument("--void-depth-min", type=float, default=0.55, help="Minimum superbubble suppression depth")
    parser.add_argument("--void-depth-max", type=float, default=0.90, help="Maximum superbubble suppression depth")
    parser.add_argument("--void-floor", type=float, default=0.05, help="Minimum multiplicative gas residual inside voids")
    parser.add_argument("--void-floor-min", type=float, default=0.02, help="Random preview minimum void floor")
    parser.add_argument("--void-floor-max", type=float, default=0.10, help="Random preview maximum void floor")
    parser.add_argument("--diffusion-sigma", type=float, default=0.075, help="Physical CR diffusion low-pass Gaussian sigma in degrees")
    parser.add_argument("--diffusion-sigma-min", type=float, default=0.05, help="Random preview minimum diffusion sigma")
    parser.add_argument("--diffusion-sigma-max", type=float, default=0.10, help="Random preview maximum diffusion sigma")
    parser.add_argument("--clip-percentile", type=float, default=99.8, help="High percentile used to robustly scale log-normal clumps")
    parser.add_argument("--clip-percentile-min", type=float, default=99.5, help="Random preview minimum clump clip percentile")
    parser.add_argument("--clip-percentile-max", type=float, default=99.9, help="Random preview maximum clump clip percentile")
    parser.add_argument("--display-gamma", type=float, default=0.62, help="PowerNorm gamma for preview display only")
    parser.add_argument("--size", type=int, default=SIM_RES, help="Image width and height in pixels")
    parser.add_argument("--fov", type=float, default=FOV_DEG, help="Field of view in degrees")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible preview")
    parser.add_argument("--save", default=None, help="Optional path to save the plotted image")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate the Type-C MC dataset")
    parser.add_argument("--dataset-count", type=int, default=1000, help="Number of images in the dataset")
    parser.add_argument("--dataset-output", default=DATASET_OUTPUT, help="Output .npy path for the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.generate_dataset:
        data = generate_type_c_mc_dataset(
            count=args.dataset_count,
            intensity_range=(args.intensity_min, args.intensity_max),
            clump_beta_range=(args.clump_beta_min, args.clump_beta_max),
            clump_alpha_range=(args.clump_alpha_min, args.clump_alpha_max),
            filament_beta_range=(args.filament_beta_min, args.filament_beta_max),
            filament_strength_range=(args.filament_strength_min, args.filament_strength_max),
            filament_power_range=(args.filament_power_min, args.filament_power_max),
            filament_octaves=args.filament_octaves,
            ridge_width_range=(args.ridge_width_min, args.ridge_width_max),
            background_floor_range=(args.background_floor_min, args.background_floor_max),
            void_count_range=(args.void_count_min, args.void_count_max),
            void_sigma_range=(args.void_sigma_min, args.void_sigma_max),
            void_depth_range=(args.void_depth_min, args.void_depth_max),
            void_floor_range=(args.void_floor_min, args.void_floor_max),
            diffusion_sigma_range=(args.diffusion_sigma_min, args.diffusion_sigma_max),
            clip_percentile_range=(args.clip_percentile_min, args.clip_percentile_max),
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
        show_type_c_mc_source(
            random=args.random,
            intensity=args.intensity,
            intensity_min=args.intensity_min,
            intensity_max=args.intensity_max,
            clump_beta=args.clump_beta,
            clump_beta_min=args.clump_beta_min,
            clump_beta_max=args.clump_beta_max,
            clump_alpha=args.clump_alpha,
            clump_alpha_min=args.clump_alpha_min,
            clump_alpha_max=args.clump_alpha_max,
            filament_beta=args.filament_beta,
            filament_beta_min=args.filament_beta_min,
            filament_beta_max=args.filament_beta_max,
            filament_strength=args.filament_strength,
            filament_strength_min=args.filament_strength_min,
            filament_strength_max=args.filament_strength_max,
            filament_power=args.filament_power,
            filament_power_min=args.filament_power_min,
            filament_power_max=args.filament_power_max,
            filament_octaves=args.filament_octaves,
            ridge_width=args.ridge_width,
            ridge_width_min=args.ridge_width_min,
            ridge_width_max=args.ridge_width_max,
            background_floor=args.background_floor,
            background_floor_min=args.background_floor_min,
            background_floor_max=args.background_floor_max,
            void_count=args.void_count,
            void_count_min=args.void_count_min,
            void_count_max=args.void_count_max,
            void_sigma_min=args.void_sigma_min,
            void_sigma_max=args.void_sigma_max,
            void_depth_min=args.void_depth_min,
            void_depth_max=args.void_depth_max,
            void_floor=args.void_floor,
            void_floor_min=args.void_floor_min,
            void_floor_max=args.void_floor_max,
            diffusion_sigma=args.diffusion_sigma,
            diffusion_sigma_min=args.diffusion_sigma_min,
            diffusion_sigma_max=args.diffusion_sigma_max,
            clip_percentile=args.clip_percentile,
            clip_percentile_min=args.clip_percentile_min,
            clip_percentile_max=args.clip_percentile_max,
            display_gamma=args.display_gamma,
            size=args.size,
            fov=args.fov,
            seed=args.seed,
            save=args.save,
        )
