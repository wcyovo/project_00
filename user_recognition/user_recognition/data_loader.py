# -*- coding: utf-8 -*-
"""睡姿压力数据加载与解析

原始 txt 文件格式：
- 每个文件记录一个人的一个具体动作，包含若干帧；
- 每帧 44 行，每行 24 个以逗号分隔的整数；
- 帧与帧之间以空行分隔；
- 空载 / 动态文件中可能还夹杂标签行（0/1/2），需跳过。

本模块负责：
1. 扫描所有人的动作文件，解析成 (frames, 44, 24) 的 numpy 数组；
2. 读入身高体重信息（readme.md）；
3. 提供统一数据结构（user_actions / user_info）供后续特征提取与匹配使用。
"""

import os
import re
import numpy as np

from . import config

# 需要忽略的非动作文件（空载 / 动态），文件名或内容含这些关键字。
# 空载文件是"床垫上没人"的基线读数，动态文件是多人/异常数据，
# 都不属于任何单一用户的动作，识别时不能混进来。
_IGNORE_KEYWORDS = ["空载", "动态", "static", "motion"]


def parse_single_file(path):
    """解析单个 txt 文件，返回 (n_frames, ROWS, COLS) 的 ndarray。

    解析规则：
    - 以"空行"作为帧边界：累积的非空数据行达到 44 行即构成一帧；
    - 跳过不足 24 列的行（可能是标签行 0/1/2），跳过无法转成数字的行；
    - 最终只保留正好 44 行 24 列的合法帧。
    """
    frames = []      # 存放所有解析出的帧
    rows = []        # 临时累积当前帧的数据行
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                # 遇到空行：说明当前帧结束，把累积的行保存为一帧
                if rows:
                    frames.append(np.array(rows, dtype=np.float32))
                    rows = []
                continue
            parts = line.split(",")
            # 跳过标签行 / 不足24列的行
            if len(parts) < config.COLS:
                continue
            try:
                # 只取前 COLS 列，转成浮点数（压力值是整数但统一用 float32 存储）
                row = [float(x) for x in parts[: config.COLS]]
            except ValueError:
                # 非数字内容（如空载标签、异常记录），整行跳过
                continue
            rows.append(row)
    # 文件末尾若还有未闭合的帧，也要保存
    if rows:
        frames.append(np.array(rows, dtype=np.float32))

    if not frames:
        raise ValueError(f"文件 {path} 未解析到有效数据")

    # 帧长不规整（不是 44 行）的帧属于异常数据，丢弃
    frames = [fr for fr in frames if fr.shape[0] == config.ROWS]
    if not frames:
        raise ValueError(f"文件 {path} 中没有合法帧 (44行x24列)")
    # 堆叠成 (n_frames, ROWS, COLS) 三维数组
    arr = np.stack(frames, axis=0).astype(np.float32)
    return arr


def parse_user_actions(user_dir):
    """解析某个用户的全部动作文件。

    一个动作 = 一个动作编号对应的所有帧。文件名约定为 <用户名>_<动作号>.txt，
    例如 dgs_1.txt / wc0604_10.txt，动作号是 "_" 后紧跟的数字。

    返回:
        actions: dict, {action_id(int): (n_frames, ROWS, COLS) ndarray}
    """
    actions = {}
    for fname in sorted(os.listdir(user_dir)):
        if not fname.lower().endswith(".txt"):
            continue
        # 跳过空载 / 动态等非动作文件
        if any(kw in fname for kw in _IGNORE_KEYWORDS):
            continue
        # 从文件名提取动作编号：正则 "_数字.txt" 结尾部分
        m = re.search(r"_(\d+)\.txt$", fname)
        if not m:
            continue
        action_id = int(m.group(1))
        arr = parse_single_file(os.path.join(user_dir, fname))
        actions[action_id] = arr
    return actions


def load_all_users():
    """加载全部用户数据。

    DATA_ROOT 下一层就是每个用户的文件夹（以用户名命名）。

    返回:
        user_actions: dict, {user_name: {action_id: (nf,ROWS,COLS) ndarray}}
        user_info:    dict, {user_name: {"height": int, "weight": int}}
    """
    user_actions = {}
    for name in sorted(os.listdir(config.DATA_ROOT)):
        user_dir = os.path.join(config.DATA_ROOT, name)
        if not os.path.isdir(user_dir):
            continue
        try:
            actions = parse_user_actions(user_dir)
        except Exception:
            # 个别文件夹解析失败不应中断整体流程，跳过即可
            continue
        # 只有解析出至少一个动作的用户才保留
        if actions:
            user_actions[name] = actions

    user_info = load_user_info()
    return user_actions, user_info


def load_user_info():
    """从 readme.md 解析每个用户的姓名、身高、体重。

    readme 形如：
        姓名     身高cm     体重
        dgs       178        75
    用空白切分：第一列姓名，第二列身高，第三列体重。

    返回:
        info: dict, {user_name: {"height": int, "weight": int}}
    """
    info = {}
    if not os.path.exists(config.INFO_FILE):
        return info
    with open(config.INFO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            name = parts[0]
            h = _to_int(parts[1])
            w = _to_int(parts[2])
            # 姓名、身高、体重三者齐全才算一条有效记录
            if name and h and w:
                info[name] = {"height": h, "weight": w}
    return info


def _to_int(s):
    """字符串转 int，失败返回 None。

    用正则提取第一个数字，兼容"178cm"这类带单位的写法。
    """
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None