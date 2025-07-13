import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self,
                 num_to_learn,
                 path_data,
                 inverse=False,
                 data_range=1.0,
                 remove_duplicates=False,
                 augment=False,
                 augment_angles=(30, 60, 90, 180, 270),
                 enhance=False,
                 remove_low_info=False,
                 ssim_threshold=0.9,
                 info_threshold=0.01,
                 device='cuda'):

        self.transform = transforms.ToTensor()
        self.device = torch.device(device)
        self.data = []

        if not os.path.exists(path_data):
            raise FileNotFoundError(f"❌ Data file not found: {path_data}")

        # ✅ 安全读取并转换为 float32 的标准 numpy 数组，防止 object 类型
        datas = np.load(path_data, allow_pickle=True)

        blurry_datas = np.stack([
            np.asarray(img, dtype=np.float32).copy()
            for img in datas[:, 1]
        ])
        original_datas = np.stack([
            np.asarray(img, dtype=np.float32).copy()
            for img in datas[:, 0]
        ])

        if inverse:
            idx_beg = blurry_datas.shape[0] - num_to_learn
        else:
            idx_beg = 0
        idx_end = idx_beg + num_to_learn

        for i in range(idx_beg, idx_end):
            img_blur = blurry_datas[i]
            img_orig = original_datas[i]

            img_blur = self._normalize(img_blur, data_range)
            img_orig = self._normalize(img_orig, data_range)

            self.data.append((img_blur.astype(np.float32), img_orig.astype(np.float32)))

        if remove_low_info:
            self._remove_low_info(info_threshold)

        if remove_duplicates:
            self._remove_duplicates_batch_gpu(ssim_threshold)

        if augment:
            self._augment_vectorized(augment_angles)

        if enhance:
            self._enhance_gamma_vectorized()

        # 最终转换为 torch tensor (1, H, W)，便于模型使用
        self.data = [
            (torch.from_numpy(b).unsqueeze(0), torch.from_numpy(o).unsqueeze(0))
            for b, o in self.data
        ]

    def _normalize(self, arr, data_range):
        arr_min, arr_max = arr.min(), arr.max()
        if abs(data_range - 1.0) < 1e-5:
            return (arr - arr_min) / (arr_max - arr_min + 1e-8)
        elif abs(data_range - 2.0) < 1e-5:
            return 2 * (arr - arr_min) / (arr_max - arr_min + 1e-8) - 1
        else:
            raise ValueError("datarange must be 1.0 or 2.0")

    def _remove_low_info(self, threshold):
        print("🗑️ Removing low-info images...")
        before = len(self.data)
        self.data = [(b, o) for b, o in self.data if np.mean(o) > threshold]
        print(f"✅ 保留样本数: {len(self.data)} / {before}")

    def _remove_duplicates_batch_gpu(self, threshold):
        print("🔁 Removing duplicates using batched GPU SSIM...")
        ssim_metric = SSIM(data_range=1.0).to(self.device)
        seen_blurry = []
        filtered = []
        ssim_vals = []

        pbar = tqdm(total=len(self.data), desc="SSIM 去重中 | 保留: 0 张")
        with torch.no_grad():
            for i, (bi_np, oi_np) in enumerate(self.data):
                bi = torch.from_numpy(bi_np).unsqueeze(0).unsqueeze(0).to(self.device)
                keep = True
                if seen_blurry:
                    b_batch = torch.stack(seen_blurry)
                    bi_batch = bi.repeat(len(b_batch), 1, 1, 1)
                    scores = ssim_metric(bi_batch, b_batch)
                    max_score = scores.max().item()
                    ssim_vals.append(max_score)
                    if max_score > threshold:
                        keep = False
                if keep:
                    seen_blurry.append(bi.squeeze(0))
                    filtered.append((bi_np, oi_np))
                    pbar.set_description(f"SSIM 去重中 | 保留: {len(filtered)} 张")
                pbar.update(1)
        pbar.close()
        avg_ssim = np.mean(ssim_vals) if ssim_vals else 0.0
        print(f"✅ 平均 SSIM: {avg_ssim:.4f} | 保留样本数: {len(filtered)} / {len(self.data)}")
        self.data = filtered

    def _augment_vectorized(self, angles):
        print("🔄 Applying vectorized data augmentation...")
        new_data = []
        for angle in angles:
            for b_np, o_np in self.data:
                b = TF.rotate(torch.from_numpy(b_np).unsqueeze(0), angle)
                o = TF.rotate(torch.from_numpy(o_np).unsqueeze(0), angle)
                new_data.append((b.squeeze(0).numpy(), o.squeeze(0).numpy()))
        print(f"✅ 扩充样本数: {len(new_data)}")
        self.data.extend(new_data)

    def _enhance_gamma_vectorized(self, gamma=1.5):
        print("⚡ Applying vectorized gamma correction...")
        self.data = [(b, np.clip(np.power(o, gamma), 0, 1)) for b, o in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
