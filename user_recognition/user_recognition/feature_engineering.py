# -*- coding: utf-8 -*-
"""数据预处理与特征增强

用户识别所需的"特征"设计要点：
- 压力图的绝对强度（总压力量级）与体重强相关，因此不能先把整体归一化去掉；
  先保留绝对强度，仅在必要时使用"整体尺度"信息。
- 睡姿差异远大于用户差异，单纯线性特征 + 最近邻在原始空间无法把用户分开，
  因此需要训练一个判别性嵌入网络（见 embedder.py），本模块只负责基础预处理与增强。

数据增强策略（题目要求"在原始数据基础上进行增强产生新的数据集"）：
由于最终数据集本身已含左右翻转（说明文档明确提示不要重复使用对称增强），
这里采用与对称无关的增强（对每一帧施加全部随机扰动中的一部分）：
1. 轻微高斯噪声（模拟传感器噪声）
2. 小范围空间平移（模拟身体微小移动）
3. 压力整体增益波动（模拟不同时刻接触面差异）
4. 随机遮挡/缺失（模拟个别传感器失效）
"""

import numpy as np

from . import config


# ------------------------- 基础预处理 -------------------------

def frame_to_vector(frame):
    """将一帧 (44,24) 压平为一维特征向量（保留绝对强度）。

    不做归一化：绝对强度 = 体重信息，是区分用户的关键信号之一，
    过早归一化会把"谁更重"这个维度抹掉。
    """
    return frame.astype(np.float32).reshape(-1)


def normalize_contrast(frame):
    """对比度归一化：整体减去空载/背景后按最大值缩放。

    注意：这里不做除以总和的全局归一化，以保留体重大小信息。
    减去最小值可去掉传感器的固定背景底噪，除以最大值保证量纲一致。
    """
    f = frame.astype(np.float32)
    f = f - float(f.min())  # 去除背景
    mx = float(f.max())
    if mx > 1e-6:
        f = f / mx
    return f


# ------------------------- 数据增强 -------------------------

def add_noise(frame, sigma=2.0, rng=None):
    """加性高斯噪声。

    模拟传感器电路噪声：每个像素叠加均值为 0、标准差为 sigma 的随机值。
    """
    rng = rng if rng is not None else np.random.default_rng()
    return frame + rng.normal(0, sigma, size=frame.shape)


def shift(frame, max_dr=1, max_dc=1, rng=None):
    """小范围平移（空位移补 0）。

    模拟人在床垫上的微小移动：整帧上下左右最多平移 1 个传感器位，
    移出的区域补 0。因为平移量很小，不会改变睡姿本质，只增加平移鲁棒性。
    """
    rng = rng if rng is not None else np.random.default_rng()
    dr = int(rng.integers(-max_dr, max_dr + 1))   # 随机行偏移
    dc = int(rng.integers(-max_dc, max_dc + 1))   # 随机列偏移
    out = np.zeros_like(frame)
    # 逐元素搬运（帧只有 44x24，双循环开销可接受）
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            ni, nj = i + dr, j + dc
            if 0 <= ni < frame.shape[0] and 0 <= nj < frame.shape[1]:
                out[ni, nj] = frame[i, j]
    return out


def gain(frame, lo=0.9, hi=1.1, rng=None):
    """压力整体增益波动。

    整体乘一个随机系数，模拟：
    - 穿着衣物厚度不同导致的压力传递差异；
    - 床垫充气压力变化导致读数整体偏大或偏小的工况。
    """
    rng = rng if rng is not None else np.random.default_rng()
    g = float(rng.uniform(lo, hi))
    return frame * g


def dropout(frame, ratio=0.05, rng=None):
    """随机置零部分传感器点位（模拟传感器失效/坏死点）。

    ratio=0.05 表示约 5% 的传感器点失效——真实床垫确实会有个别失灵传感点。
    """
    rng = rng if rng is not None else np.random.default_rng()
    mask = rng.random(frame.shape) > ratio   # True 处保留、False 处置零
    return frame * mask


def augment_frame(frame, rng=None):
    """组合增强：以一定概率依次应用噪声/平移/增益/遮挡。

    每种增强按概率独立触发，最终得到一张"变了但没变太多"的伪原始帧。
    这样训练样本数量可成倍扩大（配合 build_training_data 的 enhance_ratio），
    让网络对噪声、微小位移、增益和传感器失效都具备鲁棒性。
    """
    rng = rng if rng is not None else np.random.default_rng()
    out = frame.copy()
    if rng.random() < 0.7:
        out = add_noise(out, sigma=2.0, rng=rng)
    if rng.random() < 0.5:
        out = shift(out, max_dr=1, max_dc=1, rng=rng)
    if rng.random() < 0.7:
        out = gain(out, 0.92, 1.08, rng=rng)
    if rng.random() < 0.2:
        out = dropout(out, ratio=0.05, rng=rng)
    return out


# ------------------------- 数据集级增强 -------------------------

def build_enhanced_frames(user_actions, user_name, enhance_ratio=2, rng=None):
    """为用户构造增强后的帧样本集（含原始帧）。

    用于按动作/睡姿建模板时的数据扩充（本项目训练主流程用的是
    train_embedder.build_training_data，二者可互为替换）。

    输入 user_actions: {action_id: (nf,ROWS,COLS) ndarray}
    返回: list of (action_id, pose_id, frame)
    """
    rng = rng if rng is not None else np.random.default_rng()
    items = []
    for a, arr in user_actions.items():
        pose = config.ACTION_TO_POSE[a]
        # 原始帧原样保留
        for k in range(arr.shape[0]):
            items.append((a, pose, arr[k]))
        # 增强副本：数量 = 原始帧数 * (enhance_ratio - 1)
        n_aug = int(arr.shape[0] * (enhance_ratio - 1))
        if n_aug <= 0:
            continue
        # 随机抽取原帧（可重复）做增强，避免复制同一张
        idx = rng.choice(arr.shape[0], size=n_aug, replace=True)
        for k in idx:
            items.append((a, pose, augment_frame(arr[k], rng=rng)))
    return items