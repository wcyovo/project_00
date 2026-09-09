#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartBed 数据模拟器（M1）
-------------------------
回放仓库内真实的压力帧数据（如 data/raw/SAI/SAI_1.txt），
将“当前最新帧”以 JSON 实时暴露给 Unity 可视化端（HTTP 轮询）。

用法（在仓库根目录运行）：
    python SmartBed_Visualization/Tools/Simulator/pressure_simulator.py
    python SmartBed_Visualization/Tools/Simulator/pressure_simulator.py \
        --source SAI/SAI_1.txt --rate 5 --port 5001 --loop

依赖：pip install flask numpy
接口：
    GET /stream   -> 当前帧 JSON（见 docs/data-interface-protocol.md）
    GET /health   -> 健康检查
"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify

# ---- 从仓库 src/data_loader.py 对齐的常量 ----
ROWS, COLS = 44, 24
SLEEP_NOTE = ["仰卧", "俯卧", "左侧卧", "右侧卧"]
POSTURE_MAP = {
    "仰卧": [1, 2, 3, 4, 5, 6],
    "俯卧": [7, 8, 9],
    "左侧卧": [10, 11, 12, 13, 14, 15],
    "右侧卧": [16, 17, 18, 19, 20, 21],
}
NUM_TO_POSE = {}
for i, (name, nums) in enumerate(POSTURE_MAP.items()):
    for n in nums:
        NUM_TO_POSE[n] = (i, name)  # (sleepPoseIndex, sleepPosture)

# 8 个气囊默认充气量（M1 阶段固定；后续可替换为弱力区域增强算法输出）
DEFAULT_AIRBAGS = [
    {"id": 12, "level": 80},
    {"id": 13, "level": 80},
    {"id": 40, "level": 68},
    {"id": 41, "level": 62},
    {"id": 42, "level": 72},
    {"id": 64, "level": 66},
    {"id": 65, "level": 58},
    {"id": 66, "level": 64},
]


def parse_filename(filename: str):
    """从文件名提取用户与动作号。示例：SAI_1.txt -> ('SAI', 1)；SAI_1_1.txt -> ('SAI_1', 1)"""
    name_part = Path(filename).stem
    parts = name_part.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}", int(parts[-1])
    if len(parts) == 2:
        return parts[0], int(parts[1])
    return None, None


def load_frames(filepath: Path):
    """读取单个 txt，返回帧列表（每帧 44x24 float）。帧间以空行分隔。"""
    text = filepath.read_text(encoding="utf-8").strip()
    if not text:
        return []
    frames = []
    for block in text.split("\n\n"):
        rows = []
        for line in block.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            values = [float(x) for x in line.split(",") if x.strip()]
            if values:
                rows.append(values)
        if not rows:
            continue
        arr = np.array(rows)
        if arr.shape == (ROWS, COLS):
            frames.append(arr)
        else:
            try:
                frames.append(arr.reshape(ROWS, COLS))
            except ValueError:
                pass
    return frames


def normalize(frame: np.ndarray, lo=0.0, hi=300.0):
    """原始读数 0-300 归一到 0-100 kPa"""
    frame = np.asarray(frame, dtype=float)
    frame = np.clip(frame, 0.0, hi)
    return frame / hi * 100.0


def compute_metrics(frame: np.ndarray, threshold=10.0):
    flat = frame.ravel()
    return {
        "maxPressure": round(float(frame.max()), 2),
        "avgPressure": round(float(frame.mean()), 2),
        "contactIndex": round(float((flat > threshold).sum()) / flat.size, 3),
    }


class FrameProducer:
    """后台线程：按 rate 推进帧序号，保存“当前最新帧”。"""

    def __init__(self, frames, pose_index, pose_name, rate, loop):
        self.frames = frames
        self.pose_index = pose_index
        self.pose_name = pose_name
        self.rate = rate
        self.loop = loop
        self.current = None
        self.lock = threading.Lock()

    def build_message(self, idx):
        frame = normalize(self.frames[idx])
        flat = frame.ravel().tolist()
        return {
            "timestamp": time.time(),
            "frame": idx,
            "pressure": [round(v, 2) for v in flat],
            "sleepPosture": self.pose_name,
            "sleepPoseIndex": self.pose_index,
            "bodyRegions": [],  # M1 阶段为空；后续可从 body_part 结果填充
            "airbags": DEFAULT_AIRBAGS,
            "metrics": compute_metrics(frame),
        }

    def run(self):
        n = len(self.frames)
        i = 0
        while True:
            msg = self.build_message(i % n if self.loop else min(i, n - 1))
            with self.lock:
                self.current = msg
            i += 1
            time.sleep(1.0 / self.rate)

    def snapshot(self):
        with self.lock:
            return self.current


def main():
    parser = argparse.ArgumentParser(description="SmartBed 压力数据模拟器")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="data/raw 目录；默认自动定位到仓库 data/raw")
    parser.add_argument("--source", type=str, default="SAI/SAI_1.txt",
                        help="要回放的文件（相对 data-dir），如 SAI/SAI_1.txt")
    parser.add_argument("--rate", type=float, default=5.0, help="帧率 Hz（默认 5）")
    parser.add_argument("--port", type=int, default=5001, help="监听端口（默认 5001）")
    parser.add_argument("--loop", action="store_true", help="到达末帧后循环回放")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # 脚本位于 SmartBed_Visualization/Tools/Simulator/，仓库根 = parents[3]
        data_dir = Path(__file__).resolve().parents[3] / "data" / "raw"

    filepath = data_dir / args.source
    if not filepath.exists():
        print(f"[错误] 找不到数据文件：{filepath}")
        print(f"       请确认 --data-dir / --source 正确，或先执行 git pull 拉取 data/raw。")
        return

    frames = load_frames(filepath)
    if not frames:
        print(f"[错误] {filepath} 未解析出任何帧")
        return
    print(f"[信息] 已加载 {len(frames)} 帧，尺寸 {frames[0].shape}（{ROWS}x{COLS}）")

    name, action = parse_filename(filepath.name)
    pose_index, pose_name = NUM_TO_POSE.get(action, (0, "仰卧"))
    print(f"[信息] 用户={name} 动作={action} 睡姿={pose_name}({pose_index})")

    producer = FrameProducer(frames, pose_index, pose_name, args.rate, args.loop)
    t = threading.Thread(target=producer.run, daemon=True)
    t.start()

    app = Flask(__name__)

    @app.route("/stream")
    def stream():
        msg = producer.snapshot()
        if msg is None:
            return jsonify({"error": "尚未生成数据"}), 503
        return jsonify(msg)

    @app.route("/health")
    def health():
        return jsonify({"ok": True, "frames": len(frames), "rate": args.rate})

    print(f"[信息] 模拟器已启动：http://127.0.0.1:{args.port}/stream")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
