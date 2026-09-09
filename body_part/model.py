import torch
import torch.nn as nn

class BodyPartRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN特征提取器
        self.cnn = nn.Sequential(
            # 输入: (batch, 1, 44, 24)
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 22x12
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), # 11x6
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), # 5x3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), # 2x1
            nn.Flatten()  # 256 * 2 * 1 = 512
        )
        
        # 睡姿嵌入：4种睡姿 → 8维向量
        self.pose_embedding = nn.Embedding(4, 8)
        
        # 全连接回归头
        self.fc = nn.Sequential(
            nn.Linear(512 + 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 20)  # 5个部位 × 4个坐标
        )
    
    def forward(self, pressure, sleep_pose):
        """
        参数:
            pressure: (batch, 1, 44, 24)
            sleep_pose: (batch,)
        返回:
            (batch, 20) 归一化坐标
        """
        cnn_feat = self.cnn(pressure)           # (batch, 512)
        pose_vec = self.pose_embedding(sleep_pose)  # (batch, 8)
        combined = torch.cat([cnn_feat, pose_vec], dim=1)  # (batch, 520)
        return self.fc(combined)                # (batch, 20)