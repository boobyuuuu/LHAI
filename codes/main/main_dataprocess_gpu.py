# main_dataprocess_gpu.py

# ---- 1-1 Libraries for Path and Logging ----
from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
import sys
ADDR_ROOT = Path(__file__).resolve().parents[2]
logger.success(f"ADDR_ROOT path is: {ADDR_ROOT}")
ADDR_CODE = Path(__file__).resolve().parents[1]
sys.path.append(str(ADDR_ROOT))
logger.success(f"ADDR_CODE path is: {ADDR_CODE}")

# ---- 1-2 Libraries for dataprocess ----
from codes.function.dataprocess_gpu import ImageDataset
import numpy as np
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="🔧 图像数据预处理（GPU加速）")
    parser.add_argument('--input_path', type=str, required=True, help="原始数据路径，例如 data/xxx.npy")
    parser.add_argument('--output_path', type=str, required=True, help="处理后数据保存路径，例如 data/processed_xxx.npy")
    parser.add_argument('--num_to_learn', type=int, default=10000, help="读取样本数量")
    parser.add_argument('--data_range', type=float, default=1.0, choices=[1.0, 2.0], help="归一化范围：1.0 或 2.0")
    parser.add_argument('--inverse', action='store_true', help="是否从数据尾部加载样本")

    # 预处理选项
    parser.add_argument('--remove_duplicates', action='store_true', help="是否去除重复图像")
    parser.add_argument('--augment', action='store_true', help="是否进行图像扩充")
    parser.add_argument('--enhance', action='store_true', help="是否进行对比度增强（Gamma）")
    parser.add_argument('--remove_low_info', action='store_true', help="是否移除信息量低的图像")

    # 设备控制
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda', 'cpu'],
                        help="运行设备：cuda 或 cpu")

    args = parser.parse_args()

    print(f"\n🚀 正在使用设备：{args.device}")
    print("📦 正在加载并处理数据...")

    dataset = ImageDataset(
        num_to_learn=args.num_to_learn,
        path_data=args.input_path,
        inverse=args.inverse,
        data_range=args.data_range,
        remove_duplicates=args.remove_duplicates,
        augment=args.augment,
        enhance=args.enhance,
        remove_low_info=args.remove_low_info,
        device=args.device
    )

    print(f"✅ 预处理完成，共 {len(dataset)} 个样本")

    # ✅ 保存为 shape=(N, 2, H, W)，dtype=float32 的标准张量格式
    saved_data = []
    for blurry_tensor, original_tensor in dataset:
        sample = torch.stack([
            original_tensor.squeeze(0),  # [H, W]
            blurry_tensor.squeeze(0)     # [H, W]
        ], dim=0)  # shape: [2, H, W]
        saved_data.append(sample)
    
    saved_data_np = torch.stack(saved_data).numpy().astype(np.float32)  # [N, 2, H, W]
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, saved_data_np)


    print(f"💾 数据已保存到: {output_path.resolve()}\n")

if __name__ == '__main__':
    main()
