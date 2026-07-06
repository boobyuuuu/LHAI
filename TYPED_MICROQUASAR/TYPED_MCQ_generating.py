import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


SIM_RES = 128
FOV_DEG = 6.4
PIXEL_SCALE_DEG = FOV_DEG / SIM_RES
INFERNO = LinearSegmentedColormap.from_list(
    "mcq_jet_inferno",
    ["#000000", "#2D005C", "#8B1A5B", "#E85D04", "#FFE66D", "#FFFFFF"],
)
DATASET_OUTPUT = "Type_A_MCQ_1000_128_GT.npy"


def get_log_uniform_intensity(vmin, vmax, rng):
    return 10 ** rng.uniform(np.log10(vmin), np.log10(vmax))


def make_coordinate_grid(size=SIM_RES, fov=FOV_DEG):
    pixel_size = fov / size
    axis = np.linspace(-fov / 2, fov / 2, size, endpoint=False) + pixel_size / 2
    return np.meshgrid(axis.astype(np.float32), axis.astype(np.float32))


def get_gaussian_center(center_mu=0.0, center_sigma=0.35, max_offset=2.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    while True:
        cx = rng.normal(center_mu, center_sigma)
        cy = rng.normal(center_mu, center_sigma)
        if abs(cx) < max_offset and abs(cy) < max_offset:
            return cx, cy


def make_fractal_noise(size=SIM_RES, beta=2.8, rng=None):
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


def rotate_to_jet_frame(x, y, cx, cy, theta):
    dx = x - cx
    dy = y - cy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    u = dx * cos_t + dy * sin_t
    v = -dx * sin_t + dy * cos_t
    return u, v


def make_curved_side_components(
    u,
    v,
    side,
    jet_length=1.2,
    bend_amplitude=0.18,
    jet_width=0.045,
    lobe_width=0.22,
    lobe_length=0.34,
    hotspot_sigma=0.10,
    hotspot_aspect=1.7,
    jet_fade=0.55,
):
    s = side * u
    
    # 弯曲逻辑保持不变，依然使用 clip 保证过了喷流长度后不再乱弯
    s_clip = np.clip(s / jet_length, 0.0, 1.0)
    bend = side * bend_amplitude * np.sin(0.5 * np.pi * s_clip)
    v_rel = v - bend

    # 1. 喷流主轴 (Jet Spine)
    # 取消硬截断。为了防止喷流向后方 (s < 0) 溢出，使用 tanh 做一个极其平滑的软门控
    smooth_origin_gate = 0.5 * (1.0 + np.tanh(s / (jet_width * 2.0)))
    jet_spine = np.exp(-0.5 * (v_rel / jet_width) ** 2) * np.exp(-s / jet_fade) * smooth_origin_gate

    # 2. 扩散瓣 (Lobe)
    # 彻底取消任何门控！它是一个二维高斯，理应在所有方向上自然、平滑地衰减
    lobe_center = 0.72 * jet_length
    lobe = np.exp(
        -0.5 * ((s - lobe_center) / lobe_length) ** 2
        -0.5 * (v_rel / lobe_width) ** 2
    )

    # 3. 激波热斑 (Hotspot)
    hotspot = np.exp(
        -0.5 * ((s - jet_length) / (hotspot_sigma * hotspot_aspect)) ** 2
        -0.5 * (v_rel / hotspot_sigma) ** 2
    )
    # 以前是 hotspot * (s > 0.45 * jet_length)，现在改为软门控
    hotspot_gate = 0.5 * (1.0 + np.tanh((s - 0.45 * jet_length) / hotspot_sigma))
    hotspot = hotspot * hotspot_gate

    # 4. 前导帽 (Leading Cap)
    leading_cap = np.exp(
        -0.5 * ((s - 1.03 * jet_length) / (hotspot_sigma * 0.8)) ** 2
        -0.5 * (v_rel / (hotspot_sigma * 1.35)) ** 2
    )
    cap_gate = 0.5 * (1.0 + np.tanh((s - 0.60 * jet_length) / hotspot_sigma))
    leading_cap = leading_cap * cap_gate

    return jet_spine.astype(np.float32), lobe.astype(np.float32), (hotspot + 0.45 * leading_cap).astype(np.float32)


def apply_plasma_turbulence(image, alpha=0.28, beta=2.8, rng=None):
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


def make_gaussian_core(x, y, cx, cy, sigma=0.045):
    return np.exp(-0.5 * (((x - cx) / sigma) ** 2 + ((y - cy) / sigma) ** 2)).astype(np.float32)


def simulate_mcq_double_lobe(
    center_mu=0.0,
    center_sigma=0.35,
    max_offset=2.0,
    intensity=120.0,
    jet_length=1.2,
    theta=None,
    brightness_ratio=5.0,
    approaching_side=None,
    bend_amplitude=0.18,
    jet_width=0.045,
    lobe_width=0.22,
    lobe_length=0.34,
    hotspot_sigma=0.10,
    hotspot_aspect=1.7,
    jet_fade=0.55,
    jet_fraction=0.22,
    lobe_fraction=0.48,
    hotspot_fraction=0.30,
    core_fraction=0.04,
    turbulence_alpha=0.28,
    turbulence_beta=2.8,
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
    if approaching_side is None:
        approaching_side = 1 if rng.random() < 0.5 else -1

    u, v = rotate_to_jet_frame(x=x, y=y, cx=cx, cy=cy, theta=theta)
    image = np.zeros_like(x, dtype=np.float32)

    for side in (-1, 1):
        jet, lobe, hotspot = make_curved_side_components(
            u=u,
            v=v,
            side=side,
            jet_length=jet_length,
            bend_amplitude=bend_amplitude,
            jet_width=jet_width,
            lobe_width=lobe_width,
            lobe_length=lobe_length,
            hotspot_sigma=hotspot_sigma,
            hotspot_aspect=hotspot_aspect,
            jet_fade=jet_fade,
        )
        boost = 1.0 if side == approaching_side else 1.0 / brightness_ratio
        side_image = jet_fraction * jet + lobe_fraction * lobe + hotspot_fraction * hotspot
        image = image + boost * side_image

    image = apply_plasma_turbulence(
        image=image,
        alpha=turbulence_alpha,
        beta=turbulence_beta,
        rng=rng,
    )

    if core_fraction > 0:
        core = make_gaussian_core(x=x, y=y, cx=cx, cy=cy, sigma=max(jet_width, PIXEL_SCALE_DEG))
        image = image + core_fraction * core

    image = normalize_intensity(image, intensity)

    params = {
        "type": "A_MCQ",
        "cx": cx,
        "cy": cy,
        "jet_length": jet_length,
        "theta": theta,
        "brightness_ratio": brightness_ratio,
        "approaching_side": approaching_side,
        "bend_amplitude": bend_amplitude,
        "jet_width": jet_width,
        "lobe_width": lobe_width,
        "lobe_length": lobe_length,
        "hotspot_sigma": hotspot_sigma,
        "turbulence_alpha": turbulence_alpha,
        "turbulence_beta": turbulence_beta,
        "intensity": intensity,
    }
    return image, params


def simulate_random_type_a_mcq(
    center_mu=0.0,
    center_sigma=0.35,
    max_offset=2.0,
    intensity_range=(10, 500),
    jet_length_range=(0.55, 1.55),
    brightness_ratio_range=(2.0, 10.0),
    bend_amplitude_range=(0.04, 0.30),
    jet_width_range=(0.025, 0.075),
    lobe_width_range=(0.12, 0.34),
    lobe_length_range=(0.20, 0.48),
    hotspot_sigma_range=(0.055, 0.14),
    hotspot_aspect_range=(1.2, 2.4),
    jet_fade_range=(0.35, 0.85),
    core_fraction_range=(0.0, 0.08),
    turbulence_alpha_range=(0.12, 0.42),
    turbulence_beta_range=(2.2, 3.6),
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
):
    rng = np.random.default_rng(seed)
    return simulate_mcq_double_lobe(
        center_mu=center_mu,
        center_sigma=center_sigma,
        max_offset=max_offset,
        intensity=get_log_uniform_intensity(*intensity_range, rng=rng),
        jet_length=rng.uniform(*jet_length_range),
        theta=rng.uniform(0, np.pi),
        brightness_ratio=rng.uniform(*brightness_ratio_range),
        approaching_side=1 if rng.random() < 0.5 else -1,
        bend_amplitude=rng.uniform(*bend_amplitude_range),
        jet_width=rng.uniform(*jet_width_range),
        lobe_width=rng.uniform(*lobe_width_range),
        lobe_length=rng.uniform(*lobe_length_range),
        hotspot_sigma=rng.uniform(*hotspot_sigma_range),
        hotspot_aspect=rng.uniform(*hotspot_aspect_range),
        jet_fade=rng.uniform(*jet_fade_range),
        core_fraction=rng.uniform(*core_fraction_range),
        turbulence_alpha=rng.uniform(*turbulence_alpha_range),
        turbulence_beta=rng.uniform(*turbulence_beta_range),
        size=size,
        fov=fov,
        seed=rng.integers(0, np.iinfo(np.int32).max),
    )


def generate_type_a_mcq_dataset(
    count=1000,
    center_mu=0.0,
    center_sigma=0.35,
    max_offset=2.0,
    intensity_range=(10, 500),
    jet_length_range=(0.55, 1.55),
    brightness_ratio_range=(2.0, 10.0),
    bend_amplitude_range=(0.04, 0.30),
    jet_width_range=(0.025, 0.075),
    lobe_width_range=(0.12, 0.34),
    lobe_length_range=(0.20, 0.48),
    hotspot_sigma_range=(0.055, 0.14),
    hotspot_aspect_range=(1.2, 2.4),
    jet_fade_range=(0.35, 0.85),
    core_fraction_range=(0.0, 0.08),
    turbulence_alpha_range=(0.12, 0.42),
    turbulence_beta_range=(2.2, 3.6),
    size=SIM_RES,
    fov=FOV_DEG,
    output=DATASET_OUTPUT,
    seed=None,
):
    rng = np.random.default_rng(seed)
    dataset = np.empty((count, size, size), dtype=np.float32)

    for i in range(count):
        image, _ = simulate_random_type_a_mcq(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity_range=intensity_range,
            jet_length_range=jet_length_range,
            brightness_ratio_range=brightness_ratio_range,
            bend_amplitude_range=bend_amplitude_range,
            jet_width_range=jet_width_range,
            lobe_width_range=lobe_width_range,
            lobe_length_range=lobe_length_range,
            hotspot_sigma_range=hotspot_sigma_range,
            hotspot_aspect_range=hotspot_aspect_range,
            jet_fade_range=jet_fade_range,
            core_fraction_range=core_fraction_range,
            turbulence_alpha_range=turbulence_alpha_range,
            turbulence_beta_range=turbulence_beta_range,
            size=size,
            fov=fov,
            seed=rng.integers(0, np.iinfo(np.int32).max),
        )
        dataset[i] = image

    np.save(output, dataset)
    return dataset


def show_type_a_mcq_source(
    random=False,
    center_mu=0.0,
    center_sigma=0.35,
    max_offset=2.0,
    intensity=120.0,
    intensity_min=10.0,
    intensity_max=500.0,
    jet_length=1.2,
    jet_length_min=0.55,
    jet_length_max=1.55,
    theta=None,
    brightness_ratio=5.0,
    brightness_ratio_min=2.0,
    brightness_ratio_max=10.0,
    bend_amplitude=0.18,
    bend_amplitude_min=0.04,
    bend_amplitude_max=0.30,
    jet_width=0.045,
    jet_width_min=0.025,
    jet_width_max=0.075,
    lobe_width=0.22,
    lobe_width_min=0.12,
    lobe_width_max=0.34,
    lobe_length=0.34,
    lobe_length_min=0.20,
    lobe_length_max=0.48,
    hotspot_sigma=0.10,
    hotspot_sigma_min=0.055,
    hotspot_sigma_max=0.14,
    hotspot_aspect=1.7,
    hotspot_aspect_min=1.2,
    hotspot_aspect_max=2.4,
    jet_fade=0.55,
    jet_fade_min=0.35,
    jet_fade_max=0.85,
    core_fraction=0.04,
    core_fraction_min=0.0,
    core_fraction_max=0.08,
    turbulence_alpha=0.28,
    turbulence_alpha_min=0.12,
    turbulence_alpha_max=0.42,
    turbulence_beta=2.8,
    turbulence_beta_min=2.2,
    turbulence_beta_max=3.6,
    size=SIM_RES,
    fov=FOV_DEG,
    seed=None,
    save=None,
):
    if random:
        image, params = simulate_random_type_a_mcq(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity_range=(intensity_min, intensity_max),
            jet_length_range=(jet_length_min, jet_length_max),
            brightness_ratio_range=(brightness_ratio_min, brightness_ratio_max),
            bend_amplitude_range=(bend_amplitude_min, bend_amplitude_max),
            jet_width_range=(jet_width_min, jet_width_max),
            lobe_width_range=(lobe_width_min, lobe_width_max),
            lobe_length_range=(lobe_length_min, lobe_length_max),
            hotspot_sigma_range=(hotspot_sigma_min, hotspot_sigma_max),
            hotspot_aspect_range=(hotspot_aspect_min, hotspot_aspect_max),
            jet_fade_range=(jet_fade_min, jet_fade_max),
            core_fraction_range=(core_fraction_min, core_fraction_max),
            turbulence_alpha_range=(turbulence_alpha_min, turbulence_alpha_max),
            turbulence_beta_range=(turbulence_beta_min, turbulence_beta_max),
            size=size,
            fov=fov,
            seed=seed,
        )
    else:
        image, params = simulate_mcq_double_lobe(
            center_mu=center_mu,
            center_sigma=center_sigma,
            max_offset=max_offset,
            intensity=intensity,
            jet_length=jet_length,
            theta=theta,
            brightness_ratio=brightness_ratio,
            bend_amplitude=bend_amplitude,
            jet_width=jet_width,
            lobe_width=lobe_width,
            lobe_length=lobe_length,
            hotspot_sigma=hotspot_sigma,
            hotspot_aspect=hotspot_aspect,
            jet_fade=jet_fade,
            core_fraction=core_fraction,
            turbulence_alpha=turbulence_alpha,
            turbulence_beta=turbulence_beta,
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
    ax.set_title("Type A MCQ Double-lobe Jet Preview")
    ax.set_xlabel("RA offset (deg)")
    ax.set_ylabel("Dec offset (deg)")

    ax.text(
        0.02,
        0.98,
        f"{params['type']}\n"
        f"cx={params['cx']:.3f}°, cy={params['cy']:.3f}°\n"
        f"L={params['jet_length']:.2f}°, theta={params['theta']:.2f}, ratio={params['brightness_ratio']:.1f}\n"
        f"bend={params['bend_amplitude']:.2f}°, jetw={params['jet_width']:.3f}°, lobew={params['lobe_width']:.2f}°\n"
        f"hot={params['hotspot_sigma']:.2f}°, turb={params['turbulence_alpha']:.2f}, side={params['approaching_side']}\n"
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
    parser = argparse.ArgumentParser(description="Preview Type-A MCQ double-lobe jet generation.")
    parser.add_argument("--random", action="store_true", help="Randomly sample Type-A MCQ physical parameters")
    parser.add_argument("--center-mu", type=float, default=0.0, help="Gaussian center distribution mean in degrees")
    parser.add_argument("--center-sigma", type=float, default=0.35, help="Gaussian center distribution sigma in degrees")
    parser.add_argument("--max-offset", type=float, default=2.0, help="Maximum absolute center offset in degrees")
    parser.add_argument("--intensity", type=float, default=120.0, help="Total source intensity for non-random preview")
    parser.add_argument("--intensity-min", type=float, default=10.0, help="Random preview minimum total intensity")
    parser.add_argument("--intensity-max", type=float, default=500.0, help="Random preview maximum total intensity")
    parser.add_argument("--jet-length", type=float, default=1.2, help="One-sided jet length in degrees")
    parser.add_argument("--jet-length-min", type=float, default=0.55, help="Random preview minimum one-sided jet length")
    parser.add_argument("--jet-length-max", type=float, default=1.55, help="Random preview maximum one-sided jet length")
    parser.add_argument("--theta", type=float, default=None, help="Jet axis angle in radians")
    parser.add_argument("--brightness-ratio", type=float, default=3.0, help="Approaching/receding side brightness ratio")
    parser.add_argument("--brightness-ratio-min", type=float, default=2.0, help="Random preview minimum brightness ratio")
    parser.add_argument("--brightness-ratio-max", type=float, default=8.0, help="Random preview maximum brightness ratio")
    parser.add_argument("--bend-amplitude", type=float, default=0.04, help="S-shaped precession bending amplitude in degrees")
    parser.add_argument("--bend-amplitude-min", type=float, default=0.04, help="Random preview minimum bending amplitude")
    parser.add_argument("--bend-amplitude-max", type=float, default=0.30, help="Random preview maximum bending amplitude")
    parser.add_argument("--jet-width", type=float, default=0.045, help="Narrow jet spine width in degrees")
    parser.add_argument("--jet-width-min", type=float, default=0.025, help="Random preview minimum jet spine width")
    parser.add_argument("--jet-width-max", type=float, default=0.075, help="Random preview maximum jet spine width")
    parser.add_argument("--lobe-width", type=float, default=0.22, help="Diffuse lobe transverse width in degrees")
    parser.add_argument("--lobe-width-min", type=float, default=0.12, help="Random preview minimum lobe width")
    parser.add_argument("--lobe-width-max", type=float, default=0.34, help="Random preview maximum lobe width")
    parser.add_argument("--lobe-length", type=float, default=0.34, help="Diffuse lobe longitudinal width in degrees")
    parser.add_argument("--lobe-length-min", type=float, default=0.20, help="Random preview minimum lobe length")
    parser.add_argument("--lobe-length-max", type=float, default=0.48, help="Random preview maximum lobe length")
    parser.add_argument("--hotspot-sigma", type=float, default=0.10, help="Hotspot transverse sigma in degrees")
    parser.add_argument("--hotspot-sigma-min", type=float, default=0.055, help="Random preview minimum hotspot sigma")
    parser.add_argument("--hotspot-sigma-max", type=float, default=0.14, help="Random preview maximum hotspot sigma")
    parser.add_argument("--hotspot-aspect", type=float, default=1.7, help="Hotspot elongation along jet direction")
    parser.add_argument("--hotspot-aspect-min", type=float, default=1.2, help="Random preview minimum hotspot aspect")
    parser.add_argument("--hotspot-aspect-max", type=float, default=2.4, help="Random preview maximum hotspot aspect")
    parser.add_argument("--jet-fade", type=float, default=0.55, help="Exponential jet spine fading scale in degrees")
    parser.add_argument("--jet-fade-min", type=float, default=0.35, help="Random preview minimum jet fading scale")
    parser.add_argument("--jet-fade-max", type=float, default=0.85, help="Random preview maximum jet fading scale")
    parser.add_argument("--core-fraction", type=float, default=0.04, help="Weak central core component weight before normalization")
    parser.add_argument("--core-fraction-min", type=float, default=0.0, help="Random preview minimum central core fraction")
    parser.add_argument("--core-fraction-max", type=float, default=0.08, help="Random preview maximum central core fraction")
    parser.add_argument("--turbulence-alpha", type=float, default=0.28, help="Plasma turbulence modulation amplitude")
    parser.add_argument("--turbulence-alpha-min", type=float, default=0.12, help="Random preview minimum turbulence amplitude")
    parser.add_argument("--turbulence-alpha-max", type=float, default=0.42, help="Random preview maximum turbulence amplitude")
    parser.add_argument("--turbulence-beta", type=float, default=2.8, help="Plasma turbulence fractal spectral index")
    parser.add_argument("--turbulence-beta-min", type=float, default=2.2, help="Random preview minimum turbulence beta")
    parser.add_argument("--turbulence-beta-max", type=float, default=3.6, help="Random preview maximum turbulence beta")
    parser.add_argument("--size", type=int, default=SIM_RES, help="Image width and height in pixels")
    parser.add_argument("--fov", type=float, default=FOV_DEG, help="Field of view in degrees")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible preview")
    parser.add_argument("--save", default=None, help="Optional path to save the plotted image")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate the Type-A MCQ dataset")
    parser.add_argument("--dataset-count", type=int, default=1000, help="Number of images in the dataset")
    parser.add_argument("--dataset-output", default=DATASET_OUTPUT, help="Output .npy path for the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.generate_dataset:
        data = generate_type_a_mcq_dataset(
            count=args.dataset_count,
            center_mu=args.center_mu,
            center_sigma=args.center_sigma,
            max_offset=args.max_offset,
            intensity_range=(args.intensity_min, args.intensity_max),
            jet_length_range=(args.jet_length_min, args.jet_length_max),
            brightness_ratio_range=(args.brightness_ratio_min, args.brightness_ratio_max),
            bend_amplitude_range=(args.bend_amplitude_min, args.bend_amplitude_max),
            jet_width_range=(args.jet_width_min, args.jet_width_max),
            lobe_width_range=(args.lobe_width_min, args.lobe_width_max),
            lobe_length_range=(args.lobe_length_min, args.lobe_length_max),
            hotspot_sigma_range=(args.hotspot_sigma_min, args.hotspot_sigma_max),
            hotspot_aspect_range=(args.hotspot_aspect_min, args.hotspot_aspect_max),
            jet_fade_range=(args.jet_fade_min, args.jet_fade_max),
            core_fraction_range=(args.core_fraction_min, args.core_fraction_max),
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
        show_type_a_mcq_source(
            random=args.random,
            center_mu=args.center_mu,
            center_sigma=args.center_sigma,
            max_offset=args.max_offset,
            intensity=args.intensity,
            intensity_min=args.intensity_min,
            intensity_max=args.intensity_max,
            jet_length=args.jet_length,
            jet_length_min=args.jet_length_min,
            jet_length_max=args.jet_length_max,
            theta=args.theta,
            brightness_ratio=args.brightness_ratio,
            brightness_ratio_min=args.brightness_ratio_min,
            brightness_ratio_max=args.brightness_ratio_max,
            bend_amplitude=args.bend_amplitude,
            bend_amplitude_min=args.bend_amplitude_min,
            bend_amplitude_max=args.bend_amplitude_max,
            jet_width=args.jet_width,
            jet_width_min=args.jet_width_min,
            jet_width_max=args.jet_width_max,
            lobe_width=args.lobe_width,
            lobe_width_min=args.lobe_width_min,
            lobe_width_max=args.lobe_width_max,
            lobe_length=args.lobe_length,
            lobe_length_min=args.lobe_length_min,
            lobe_length_max=args.lobe_length_max,
            hotspot_sigma=args.hotspot_sigma,
            hotspot_sigma_min=args.hotspot_sigma_min,
            hotspot_sigma_max=args.hotspot_sigma_max,
            hotspot_aspect=args.hotspot_aspect,
            hotspot_aspect_min=args.hotspot_aspect_min,
            hotspot_aspect_max=args.hotspot_aspect_max,
            jet_fade=args.jet_fade,
            jet_fade_min=args.jet_fade_min,
            jet_fade_max=args.jet_fade_max,
            core_fraction=args.core_fraction,
            core_fraction_min=args.core_fraction_min,
            core_fraction_max=args.core_fraction_max,
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
