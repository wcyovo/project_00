import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from scipy.ndimage import rotate

JSON_PATH = 'data_2026.json'

class BodyPartDataset(Dataset):
    def __init__(self, json_path, train=True, train_ratio=0.7):
        self.train = train

        with open(json_path, 'r', encoding='utf-8') as f:
            self.all_samples = json.load(f)

        # ===== 按睡姿分层随机划分 =====
        np.random.seed(42)

        # 按睡姿分组
        pose_groups = {}
        for s in self.all_samples:
            pose = s['sleep_pos']
            pose_groups.setdefault(pose, []).append(s)

        train_samples = []
        test_samples = []

        for pose, samples in pose_groups.items():
            np.random.shuffle(samples)
            split = int(len(samples) * train_ratio)
            train_samples.extend(samples[:split])
            test_samples.extend(samples[split:])

        self.samples = train_samples if train else test_samples

        print(f"{'训练' if train else '验证'}集: {len(self.samples)} 条样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        data_vals = [float(x) for x in s['data'].split(',')]
        pressure = np.array(data_vals, dtype=np.float32).reshape(44, 24)

        # ===== 数据增强（训练集） =====
        if self.train:
            # 噪声
            noise = np.random.randn(44, 24) * 0.02 * (np.max(pressure) + 1e-6)
            pressure = pressure + noise
            pressure = np.clip(pressure, 0, None)

            # 旋转（稍微加大）
            angle = random.uniform(-5, 5)
            pressure = rotate(pressure, angle, reshape=False, order=1)
            pressure = np.clip(pressure, 0, None)

            # 平移
            shift_x = random.randint(-2, 2)
            shift_y = random.randint(-2, 2)
            pressure = np.roll(pressure, shift_x, axis=1)
            pressure = np.roll(pressure, shift_y, axis=0)

        # region 解析
        region_vals = s['region'].split()
        x_coords = [float(v) for v in region_vals[:10]]
        y_coords = [float(v) for v in region_vals[12:22]]

        target = []
        for i in range(5):
            target.extend([x_coords[i*2], y_coords[i*2], x_coords[i*2+1], y_coords[i*2+1]])
        target = np.array(target, dtype=np.float32)

        for i in range(0, 20, 2):
            target[i] = target[i] / 24.0
            target[i+1] = target[i+1] / 44.0

        return {
            'pressure': torch.FloatTensor(pressure).unsqueeze(0),
            'target': torch.FloatTensor(target),
            'sleep_pose': torch.LongTensor([s['sleep_pos']]),
            'people_name': s['people_name']
        }