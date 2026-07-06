import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


SIM_RES = 128
FOV_DEG = 6.4
PIXEL_SCALE_DEG = FOV_DEG / SIM_RES
THICKNESS = 2.0
AFMHOT = LinearSegmentedColormap.from_list(
    "hess_afmhot",
    ["#000000", "#800000", "#FF4500", "#FFFF00", "#FFFFFF"],
)
DATASET_OUTPUT = "Type_C_SHELL_1000_128_GT.npy"


def generate_base_noise(size=SIM_RES, seed=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    return rng.random((size, size), dtype=np.float32)


def _box_mean_axis(data, radius, axis):
    moved = np.moveaxis(data, axis, -1)
    zeros = np.zeros((*moved.shape[:-1], 1), dtype=moved.dtype)
    cumsum = np.concatenate([zeros, np.cumsum(moved, axis=-1)], axis=-1)

    n = moved.shape[-1]
    idx = np.arange(n)
    start = np.maximum(0, idx - radius)
    stop = np.minimum(n, idx + radius + 1)
    mean = (np.take(cumsum, stop, axis=-1) - np.take(cumsum, start, axis=-1)) / (
        stop - start
    )
    return np.moveaxis(mean, -1, axis).astype(np.float32, copy=False)


def apply_blur(noise, sigma):
    radius = max(1, int(np.floor(sigma * 2)))
    temp = _box_mean_axis(noise, radius, axis=1)
    return _box_mean_axis(temp, radius, axis=0)


def make_coordinate_grid(size=SIM_RES, fov=FOV_DEG):
    pixel_size = fov / size
    axis = np.linspace(-fov / 2, fov / 2, size, endpoint=False) + pixel_size / 2
    return np.meshgrid(axis.astype(np.float32), axis.astype(np.float32))


def normalize_intensity(image, intensity):
    total = image.sum(dtype=np.float64)
    if total > 0:
        image = image / total * intensity
    return image.astype(np.float32, copy=False)


def simulate_snr_shell(
    radius=15,
    thickness=THICKNESS,
    sigma=0.5,
    threshold=0.4,
    asymmetry=0.5,
    size=SIM_RES,
    seed=None,
    rng=None,
    cx_offset=0.0,  # 新增：X轴中心偏移量
    cy_offset=0.0,  # 新增：Y轴中心偏移量
):
    raw_noise = generate_base_noise(size=size, seed=seed, rng=rng)
    processed_noise = apply_blur(raw_noise, sigma)

    y, x = np.indices((size, size), dtype=np.float32)
    # 新增：将偏移量加入到中心坐标中
    cx = size / 2 + cx_offset
    cy = size / 2 + cy_offset
    dx = x - cx
    dy = y - cy
    dist = np.sqrt(dx * dx + dy * dy)

    ring_intensity = np.exp(-((dist - radius) ** 2) / (2 * thickness * thickness))
    if threshold >= 1:
        gating = np.zeros_like(processed_noise, dtype=np.float32)
    else:
        gating = np.where(
            processed_noise > threshold,
            (processed_noise - threshold) / (1 - threshold),
            0,
        )

    angle = np.arctan2(dy, dx)
    asym_factor = 1.0 - (asymmetry * (0.5 + 0.5 * np.cos(angle)))
    snr_map = np.minimum(1.0, ring_intensity * gating * asym_factor * 1.5)

    return snr_map.astype(np.float32, copy=False)


def simulate_controlled_shell(
    outer_diameter_deg,
    inner_diameter_deg,
    intensity=200.0,
    blur_sigma_deg=0.05,
    size=SIM_RES,
    fov=FOV_DEG,
):
    x, y = make_coordinate_grid(size=size, fov=fov)
    dist = np.sqrt(x * x + y * y)
    pixel_scale = fov / size
    outer_radius = outer_diameter_deg / 2
    inner_radius = inner_diameter_deg / 2
    radial_center = (outer_radius + inner_radius) / 2
    radial_half_width = max((outer_radius - inner_radius) / 2, pixel_scale / 2)
    image = (np.abs(dist - radial_center) <= radial_half_width).astype(np.float32)
    if blur_sigma_deg > 0:
        from Type_A_DISK_generating import apply_gaussian_blur

        image = apply_gaussian_blur(image, sigma_deg=blur_sigma_deg, pixel_scale_deg=pixel_scale)
    return normalize_intensity(image, intensity)


def show_snr_shell(
    radius=15,
    thickness=THICKNESS,
    sigma=0.5,
    threshold=0.4,
    asymmetry=0.5,
    size=SIM_RES,
    seed=None,
    save=None,
):
    snr_map = simulate_snr_shell(
        radius=radius,
        thickness=thickness,
        sigma=sigma,
        threshold=threshold,
        asymmetry=asymmetry,
        size=size,
        seed=seed,
    )

    active_pixels = int(np.count_nonzero(snr_map > 0.1))
    peak_flux = float(snr_map.max())

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    image = ax.imshow(snr_map, origin="lower", cmap=AFMHOT, vmin=0, vmax=1)
    ax.set_title("J1713-like HESS SNR Shell Fragmenter")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    scale_y = size - 10
    ax.plot([10, 20], [scale_y, scale_y], color="white", lw=2)
    ax.text(10, scale_y - 4, "10px (Sim Scale)", color="white", fontsize=9)

    ax.text(
        0.02,
        0.98,
        f"Frag. Count: {active_pixels // 3}\nResolution: {size}x{size}\nPeak Flux: {peak_flux:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 4},
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Normalized flux")

    if save:
        fig.savefig(save, dpi=200)
    plt.show()

    return snr_map


def generate_type_c_shell_dataset(
    count=1000,
    size=SIM_RES,
    output=DATASET_OUTPUT,
    seed=None,
    max_offset=0.0,  # 新增：最大偏移距离参数
):
    rng = np.random.default_rng(seed)
    dataset = np.empty((count, size, size), dtype=np.float32)

    for i in range(count):
        radius = rng.uniform(3, 10)
        thickness = rng.uniform(0.2 * radius, 0.75 * radius)
        sigma = rng.uniform(0, 2)
        threshold = rng.uniform(0, 0.2)
        asymmetry = rng.uniform(0, 1)
        
        # 新增：生成随机偏移
        # 使用 rng.uniform(0, 1) 开根号是为了保证在圆域内均匀分布
        # 如果你只想要距离在 [0, max_offset] 均匀分布，可以改为 offset_r = rng.uniform(0, max_offset)
        offset_r = np.sqrt(rng.uniform(0, 1)) * max_offset 
        offset_theta = rng.uniform(0, 2 * np.pi)
        cx_offset = offset_r * np.cos(offset_theta)
        cy_offset = offset_r * np.sin(offset_theta)

        dataset[i] = simulate_snr_shell(
            radius=radius,
            thickness=thickness,
            sigma=sigma,
            threshold=threshold,
            asymmetry=asymmetry,
            size=size,
            rng=rng,
            cx_offset=cx_offset,  # 新增：传入 X 偏移
            cy_offset=cy_offset,  # 新增：传入 Y 偏移
        )

    np.save(output, dataset)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate a J1713-like HESS shell image with fragmented outer structure."
    )
    parser.add_argument("--radius", type=float, default=10, help="SNR Radius (px)")
    parser.add_argument("--max-offset", type=float, default=25.0, help="Max distance (px) of shell center from image center") # 新增：最大偏移距离参数
    parser.add_argument("--thickness", type=float, default=THICKNESS, help="Shell thickness (px)")
    parser.add_argument("--sigma", type=float, default=0.0, help="Micro Noise Blur")
    parser.add_argument("--threshold", type=float, default=0.0, help="Gating Threshold")
    parser.add_argument("--asymmetry", type=float, default=0.0, help="Macro Asymmetry")
    parser.add_argument("--size", type=int, default=SIM_RES, help="Simulation image size in pixels")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible noise")
    parser.add_argument("--save", default=None, help="Optional path to save the plotted image")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate the Type-C shell dataset")
    parser.add_argument("--dataset-count", type=int, default=1000, help="Number of images in the dataset")
    parser.add_argument("--dataset-output", default=DATASET_OUTPUT, help="Output .npy path for the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.generate_dataset:
        data = generate_type_c_shell_dataset(
            count=args.dataset_count,
            size=args.size,
            output=args.dataset_output,
            seed=args.seed,
            max_offset=args.max_offset,  # 新增：传入最大偏移距离
        )
        print(
            f"Saved {args.dataset_output}: shape={data.shape}, dtype={data.dtype}, "
            f"min={data.min():.6f}, max={data.max():.6f}"
        )
    else:
        show_snr_shell(
            radius=args.radius,
            thickness=args.thickness,
            sigma=args.sigma,
            threshold=args.threshold,
            asymmetry=args.asymmetry,
            size=args.size,
            seed=args.seed,
            save=args.save,
        )
