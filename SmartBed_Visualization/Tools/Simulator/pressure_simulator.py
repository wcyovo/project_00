#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartBed 数据模拟器（M1）—— 标准库版（无需 flask/numpy）
-----------------------------------------------------------
回放仓库内真实的压力帧数据（如 data/raw/SAI/SAI_1.txt），
将“当前最新帧”以 JSON 实时暴露给 Unity 可视化端（HTTP 轮询）。

只用 Python 标准库：运行前仅需一个可用的 Python 3 解释器即可。

用法（在仓库根目录运行）：
    python SmartBed_Visualization/Tools/Simulator/pressure_simulator.py
    python SmartBed_Visualization/Tools/Simulator/pressure_simulator.py \
        --source SAI/SAI_1.txt --rate 5 --port 5001 --loop

接口：
    GET /stream   -> 当前帧 JSON（见 docs/data-interface-protocol.md）
    GET /health   -> 健康检查
"""

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---- 与仓库 src/data_loader.py 对齐的常量 ----
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
DEFAULT_AIRBAGS = [12, 13, 40, 41, 42, 64, 65, 66]


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
    """读取单个 txt，返回帧列表（每帧为 44x24 的 list[list[float]]）。帧间以空行分隔。"""
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
        flat = [v for row in rows for v in row]
        if len(flat) == ROWS * COLS:
            # 按 44x24 重塑（行优先）
            frames.append([flat[r * COLS:(r + 1) * COLS] for r in range(ROWS)])
    return frames


def normalize(value, hi=300.0):
    """原始读数 0-300 归一到 0-100 kPa"""
    return max(0.0, min(hi, value)) / hi * 100.0


def compute_metrics(frame, threshold=10.0):
    """frame 为 44x24 list[list[float]]（已归一化 0-100）"""
    total = 0.0
    count = 0
    mx = 0.0
    above = 0
    for row in frame:
        for v in row:
            if v > mx:
                mx = v
            total += v
            count += 1
            if v > threshold:
                above += 1
    return {
        "maxPressure": round(mx, 2),
        "avgPressure": round(total / count, 2) if count else 0.0,
        "contactIndex": round(above / count, 3) if count else 0.0,
    }


BODY_PARTS = ["肩", "背", "腰", "臀", "大腿"]


def derive_body_regions(frame, raw_threshold=25.0):
    """
    从压力帧派生 5 个身体部位框（肩/背/腰/臀/大腿）。
    先求接触区域边界框，再把纵向(y/行)均分为 5 段。坐标为 x∈0..24、y∈0..44。
    frame 为原始读数 0-300 的 44x24。仅作演示占位；后续替换为 body_part 模型真实输出。
    """
    ys, xs = [], []
    for r in range(ROWS):
        for c in range(COLS):
            if frame[r][c] > raw_threshold:
                ys.append(r)
                xs.append(c)
    if not ys:
        return []
    y0, y1 = min(ys), max(ys)
    x0, x1 = min(xs), max(xs)
    span = (y1 - y0 + 1) / len(BODY_PARTS)
    regions = []
    for i, name in enumerate(BODY_PARTS):
        ys_ = int(round(y0 + i * span))
        ye_ = int(round(y0 + (i + 1) * span)) - 1
        regions.append({
            "part": name,
            "x1": x0,
            "y1": max(0, ys_),
            "x2": x1,
            "y2": min(ROWS - 1, ye_),
        })
    return regions


def current_airbags(t):
    """返回随时间轻微变化的气囊充气量，模拟“调节支撑”。数值为 0-100。"""
    base = [80, 80, 68, 62, 72, 66, 58, 64]
    out = []
    for i, b in enumerate(base):
        v = b + 15 * math.sin(t * 0.5 + i * 0.7)
        out.append({"id": DEFAULT_AIRBAGS[i], "level": round(max(0.0, min(100.0, v)), 0)})
    return out


class FrameProducer:
    """后台线程：按 rate 推进帧序号，保存“当前最新帧”。"""

    def __init__(self, frames, user_name, pose_index, pose_name, rate, loop):
        self.frames = frames
        self.user_name = user_name
        self.pose_index = pose_index
        self.pose_name = pose_name
        self.rate = rate
        self.loop = loop
        self.current = None
        self.lock = threading.Lock()

    def build_message(self, idx):
        frame = self.frames[idx]
        flat = [round(normalize(v), 2) for row in frame for v in row]
        return {
            "timestamp": time.time(),
            "frame": idx,
            "pressure": flat,
            "sleepPosture": self.pose_name,
            "sleepPoseIndex": self.pose_index,
            "currentUser": self.user_name,
            "bodyRegions": derive_body_regions(frame),
            "airbags": current_airbags(time.time()),
            "metrics": compute_metrics([[normalize(v) for v in row] for row in frame]),
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
    parser = argparse.ArgumentParser(description="SmartBed 压力数据模拟器（标准库版）")
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
    print(f"[信息] 已加载 {len(frames)} 帧，尺寸 {ROWS}x{COLS}")

    name, action = parse_filename(filepath.name)
    pose_index, pose_name = NUM_TO_POSE.get(action, (0, "仰卧"))
    print(f"[信息] 用户={name} 动作={action} 睡姿={pose_name}({pose_index})")

    producer = FrameProducer(frames, name, pose_index, pose_name, args.rate, args.loop)
    threading.Thread(target=producer.run, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == "/stream":
                msg = producer.snapshot()
                if msg is None:
                    self._send({"error": "尚未生成数据"}, 503)
                else:
                    self._send(msg)
            elif self.path.split("?")[0] == "/health":
                self._send({"ok": True, "frames": len(frames), "rate": args.rate})
            else:
                self._send({"error": "未找到"}, 404)

        def log_message(self, fmt, *args):
            print(f"[server] {self.address_string()} - {fmt % args}")

    print(f"[信息] 模拟器已启动：http://127.0.0.1:{args.port}/stream")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
