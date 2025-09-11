import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class ImageDataset(Dataset):
    def __init__(self, num_to_learn,path_data,inverse=False, data_range=1.0):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data = []
        
        #path_blurry = os.path.join(root_directory, "SimulatedData", "blurred.npy")
        #path_original = os.path.join(root_directory, "SimulatedData", "original.npy")
        #path_data = os.path.join(root_directory, "SimulatedData", "rand_PSF.npy")
        
        #if not os.path.exists(path_blurry) or not os.path.exists(path_original):
        #    raise FileNotFoundError("Blurry or Original data file not found.")
        if not os.path.exists(path_data):
            raise FileNotFoundError("Blurry or Original data file not found.")
        
        #blurry_datas = np.load(path_blurry).astype(np.float32)
        #original_datas = np.load(path_original).astype(np.float32)
        datas = np.load(path_data,allow_pickle=True)#.astype(np.object)
        blurry_datas = np.stack(datas[:,1])
        original_datas = np.stack(datas[:,0])

        if inverse == False:
            idx_beg = 0;
            idx_end = num_to_learn;
        else:
            idx_beg = blurry_datas.shape[0]-num_to_learn;
            idx_end = blurry_datas.shape[0];


        for i in range(idx_beg,idx_end):
            blurry_data = blurry_datas[i]
            original_data = original_datas[i]
            
            if abs(data_range - 1.0) < 1e-5:
                img_blurry = (blurry_data - blurry_data.min()) / (blurry_data.max() - blurry_data.min())
                img_original = (original_data - original_data.min()) / (original_data.max() - original_data.min())

            elif abs(data_range - 2.0) < 1e-5:
                img_blurry = 2 * (blurry_data - blurry_data.min()) / (blurry_data.max() - blurry_data.min()) - 1
                img_original = 2 * (original_data - original_data.min()) / (original_data.max() - original_data.min()) - 1

            else:
                raise ValueError("datarange must be 1.0 or 2.0")
            
            img_blurry = Image.fromarray(img_blurry)
            img_original = Image.fromarray(img_original)

            img_blurry = self.transform(img_blurry)
            img_original = self.transform(img_original)
            
            self.data.append((img_blurry, img_original))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class DataModule:
    """
    负责从 .npy 构建 ImageDataset，并按给定 frac 连续切分为 train/test，
    返回与原逻辑完全一致的 DataLoader。
    """
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        frac: float = 0.8,
        inverse: bool = False,
        shuffle_train: bool = False,
        shuffle_test: bool = False,
        num_workers: int = 0, # 数据加载子进程数
        pin_memory: bool = False, # 是否将数据加载到 CUDA 可用的锁页内存
        drop_last: bool = False, # 是否丢弃最后一个不完整的 batch
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.frac = float(frac)
        self.inverse = bool(inverse)
        self.shuffle_train = bool(shuffle_train)
        self.shuffle_test = bool(shuffle_test)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)

        # 占位
        self.dataset = None
        self.trainset = None
        self.testset = None
        self.trainloader = None
        self.testloader = None

    def build(self):
        filetmp = np.load(self.data_path, allow_pickle=True)
        filelen = int(filetmp.shape[0])
        del filetmp

        self.dataset = ImageDataset(filelen, self.data_path, inverse=self.inverse)

        # 连续切分（保持与原代码一致，而非随机）
        train_size = int(self.frac * len(self.dataset))
        test_size  = len(self.dataset) - train_size

        train_indices = list(range(0, train_size))
        test_indices  = list(range(train_size, len(self.dataset)))

        self.trainset = Subset(self.dataset, train_indices)
        self.testset  = Subset(self.dataset, test_indices)

        # DataLoader（保持与原代码一致：默认 shuffle=False）
        self.trainloader = DataLoader(
            self.trainset,
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        self.testloader = DataLoader(
            self.testset,
            shuffle=self.shuffle_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        return self.trainloader, self.testloader

    def get_loaders(self):
        if self.trainloader is None or self.testloader is None:
            raise RuntimeError("You must call build() before get_loaders().")
        return self.trainloader, self.testloader