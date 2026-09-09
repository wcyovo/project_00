# -*- coding: utf-8 -*-
"""训练特征嵌入网络

用注册用户（训练用户）的全部帧（含增强样本）训练 softmax 分类网络，
训练完成后权重即为特征提取器。识别阶段只使用嵌入（不依赖分类头）。

评估协议遵循题目要求：
- 70% 用户作为训练（注册）用户，30% 用户作为"新用户"（测试/拒识评估）。
- 训练数据在原始数据基础上做增强（feature_engineering.augment_frame）。
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from . import config
from . import data_loader
from . import feature_engineering as fe
from .embedder import PressureEmbedder


def build_training_data(train_users, user_actions, enhance_ratio=2.0,
                        max_frames_per_action=15, rng=None):
    """构造训练样本。

    对每个训练用户，每个动作至多取 max_frames_per_action=15 帧（均匀采样，
    覆盖整个动作时间范围），再为每帧生成 enhance_ratio - 1 个增强副本，
    得到增强数据集后返回。

    返回:
        X: (n, input_dim) 原始压力帧压平向量（绝对强度，未归一化）
        y: (n,) 用户类别索引（int）
        names: 训练用户名单（顺序对应类别索引）
    """
    rng = rng if rng is not None else np.random.default_rng(config.RANDOM_SEED)
    names = sorted(train_users)                       # 类别顺序固定，保证可复现
    name2idx = {n: i for i, n in enumerate(names)}    # 用户名 -> 类别号

    vectors = []
    labels = []
    for name in names:
        actions = user_actions[name]
        idx = name2idx[name]
        for a, arr in actions.items():
            n_frames = arr.shape[0]
            # 在动作帧序列上均匀采样最多 max_frames_per_action 帧，
            # 避免帧多的动作（如俯卧60帧）在训练集中权重过大
            sel = np.linspace(0, n_frames - 1,
                              min(max_frames_per_action, n_frames)).astype(int)
            chosen = arr[sel]
            # 原始帧：直接压平入库
            for f in chosen:
                vectors.append(fe.frame_to_vector(f))
                labels.append(idx)
            # 增强副本：每个原始帧生成 enhance_ratio-1 个增强版本
            n_aug = int(chosen.shape[0] * (enhance_ratio - 1))
            if n_aug <= 0:
                continue
            aug_idx = rng.choice(chosen.shape[0], size=n_aug, replace=True)
            for k in aug_idx:
                aug = fe.augment_frame(chosen[k], rng=rng)
                vectors.append(fe.frame_to_vector(aug))
                labels.append(idx)

    X = np.asarray(vectors, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, names


def train_embedder(X, y, names, epochs=20, batch_size=128, lr=1e-3,
                   embed_dim=config.EMBED_DIM, verbose=True):
    """训练嵌入网络。返回 (net, scaler)。

    网络输入为原始 1056 维绝对强度向量。为加快收敛并稳定训练，
    输入前先用 StandardScaler 做 z-score 标准化（拟合在训练数据上，
    推理时复用同一套均方差——这就是 scaler.npz 的作用）。

    训练：交叉熵 + Adam。分类损失会逼迫网络学到"同用户帧的嵌入靠近、
    不同用户帧的嵌入分开"，这正是模板匹配所需的判别性。
    """
    scaler = StandardScaler().fit(X)          # 统计每个维度的均值/方差
    Xs = scaler.transform(X).astype(np.float32)

    net = PressureEmbedder(input_dim=X.shape[1], embed_dim=embed_dim,
                           n_classes=len(names))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.from_numpy(Xs)
    Yt = torch.from_numpy(y)
    net.train()

    n = len(Xs)
    for ep in range(epochs):
        perm = np.random.permutation(n)       # 每轮随机打乱，避免批次有偏
        total = 0.0
        for s in range(0, n, batch_size):
            bidx = perm[s:s + batch_size]
            xb, yb = Xt[bidx], Yt[bidx]
            out = net(xb)                     # (batch, n_classes) logits
            loss = F.cross_entropy(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(bidx)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"epoch {ep:2d}  loss {total / n:.4f}", flush=True)
    return net, scaler


def train_and_save(user_actions, users, model_path=None, scaler_path=None,
                   enhance_ratio=2.0, max_frames_per_action=15,
                   epochs=20, verbose=True):
    """一键流程：按 70/30 切分 -> 训练 -> 保存。

    返回 (net, scaler, train_users, test_users)：
    - train_users: 参与训练/注册的用户集合
    - test_users:  留作"新用户"用于拒识评估的用户名单
    """
    np.random.seed(config.RANDOM_SEED)        # 固定划分，保证可复现
    users = sorted(users)
    n_train = int(len(users) * config.TRAIN_USER_RATIO)
    train_users = set(np.random.choice(users, n_train, replace=False))
    test_users = [u for u in users if u not in train_users]

    if verbose:
        print(f"训练用户 {len(train_users)} 个: {sorted(train_users)}")
        print(f"测试(新)用户 {len(test_users)} 个: {test_users}", flush=True)

    t0 = time.time()
    X, y, names = build_training_data(train_users, user_actions,
                                      enhance_ratio, max_frames_per_action)
    if verbose:
        print(f"构建增强训练集: {X.shape} ({time.time()-t0:.1f}s)", flush=True)

    net, scaler = train_embedder(X, y, names, epochs=epochs, verbose=verbose)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if model_path is None:
        model_path = os.path.join(config.OUTPUT_DIR, "embedder.pt")
    if scaler_path is None:
        scaler_path = os.path.join(config.OUTPUT_DIR, "scaler.npz")

    net.save(model_path)                      # 网络权重
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)  # 标准化参数
    if verbose:
        print(f"模型已保存: {model_path}")
        print(f"scaler 已保存: {scaler_path}")
    return net, scaler, train_users, test_users