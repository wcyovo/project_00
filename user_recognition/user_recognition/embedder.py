# -*- coding: utf-8 -*-
"""特征嵌入网络

将 44x24=1056 维压力帧映射到低维（64 维）判别性嵌入空间。

训练方式：对（已有）注册用户帧做 softmax 多分类。训练完成后，
网络在倒数第二层输出一个 L2 归一化的嵌入向量——该嵌入被期望满足
"同一用户不同睡姿/帧距离近、不同用户距离远"。

为什么这样设计满足题目要求：
- "新用户加入时尽量不修改模型结构"：识别阶段不使用 softmax 分类头，
  而是把嵌入与模板库（每个注册用户的中心向量）做最近邻匹配；
  新用户只需把其帧输入同一网络得到嵌入并注册中心，无需重训网络。
- "除分类之外的其它实现方式（匹配）"：识别核心是匹配，分类只用来训练
  高质量的判别嵌入。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PressureEmbedder(nn.Module):
    """压力帧嵌入网络。

    input_dim: 默认 1056 (44*24)
    embed_dim: 嵌入维度（默认 64）
    n_classes: softmax 监督类别数（训练期使用）

    网络结构（全连接 + BatchNorm + ReLU）：
        fc1: 1056 -> 512
        fc2: 512  -> 256
        fc3: 256  -> 64    ← 这一层输出作为特征嵌入
        head(训练期): 64 -> n_classes  ← 仅用于监督，识别时废弃
    """

    def __init__(self, input_dim=1056, embed_dim=64, n_classes=None):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        # 三层特征提取：每层先线性变换、再 BatchNorm、最后 ReLU
        self.fc1 = nn.Linear(input_dim, 512)
        self.b1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.b2 = nn.BatchNorm1d(256)
        # 最后一层输出 embed_dim 维，不加 ReLU（允许正负特征）
        self.fc3 = nn.Linear(256, embed_dim)
        self.b3 = nn.BatchNorm1d(embed_dim)
        # softmax 分类头：n_classes 位概率，训练时用，识别时可裁剪
        self.head = nn.Linear(embed_dim, n_classes) if n_classes else None

    def embed(self, x):
        """返回 L2 归一化嵌入。

        输出向量的模长被归一化为 1：
        - 后续匹配仅用余弦相似度（等价于内积），计算简单；
        - 归一化消除了"模长"这一不稳定维度，只保留方向特征。
        """
        x = F.relu(self.b1(self.fc1(x)))
        x = F.dropout(x, 0.2, training=self.training)   # 训练期防过拟合
        x = F.relu(self.b2(self.fc2(x)))
        x = F.dropout(x, 0.2, training=self.training)
        x = self.b3(self.fc3(x))
        return F.normalize(x, dim=1)

    def forward(self, x):
        """前向：训练时走分类头，识别时直接返回嵌入。"""
        e = self.embed(x)
        if self.head is not None:
            return self.head(e)      # 训练：返回类别 logits
        return e                      # 推理：返回嵌入

    def predict_embed(self, x_np):
        """对 numpy 张量批量提取嵌入（推理模式）。

        :param x_np: (n, input_dim) 的 numpy 数组
        :return: (n, embed_dim) 归一化嵌入
        """
        self.eval()                   # 切到推理模式（关 Dropout / BatchNorm 用累计均方差）
        with torch.no_grad():         # 不计算梯度，省内存
            x = torch.from_numpy(np.asarray(x_np, dtype=np.float32))
            return self.embed(x).numpy()

    def save(self, path):
        """保存网络全部权重。"""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """加载网络权重并切到推理模式。"""
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()

    @classmethod
    def from_file(cls, path, input_dim=1056, embed_dim=64, n_classes=None):
        """从文件重建网络实例。"""
        net = cls(input_dim=input_dim, embed_dim=embed_dim, n_classes=n_classes)
        net.load(path)
        return net