# 用户识别算法的设计与实现

智能床垫项目 · 拓展功能模块 —— 用户识别

本模块根据人体躺在智能床垫上产生的压力分布，识别"当前使用者是哪一位用户"，
并支持**新用户（未注册用户）的检测与拒识**。

## 方案概述

采用 **判别性特征嵌入 + 模板匹配** 的实现方式，符合题目
"当有新用户加入时尽量不修改模型结构，可以采用匹配的方式"的要求。

```
原始压力帧 (44×24)
     │
     ▼
特征嵌入网络（softmax 监督训练的特征提取器，权重固定）
     │  输出 64 维 L2 归一化嵌入
     ▼
模板库：每位注册用户 = 一个中心模板向量
     │
     ▼
识别：余弦相似度最近邻匹配  +  阈值判未知
```


## 目录结构

```
用户识别算法/
├── main.py                        # 主流程入口
├── requirements.txt
├── data/                          # 数据目录
│   ├── 睡姿数据/                  # 原始压力 txt（33 用户）
│   └── readme.md                  # 用户身高体重信息
├── user_recognition/
│   ├── config.py                  # 全局配置（路径、超参数）
│   ├── data_loader.py             # 原始 txt 数据读取与解析
│   ├── feature_engineering.py     # 预处理与数据增强
│   ├── embedder.py                # 特征嵌入网络（PyTorch）
│   ├── train_embedder.py          # 训练流程
│   ├── matcher.py                 # 模板库 + 最近邻匹配 + 拒识
│   ├── evaluate.py                # 准确率 / FAR / FRR / 阈值选择
│   ├── visualize.py               # 降维区分度图、score 分布、阈值曲线
│   └── export_api.py              # JSON 数据接口导出
└── output/                        # 运行结果（模型、图、JSON 接口文件）
```

## 数据目录约定

代码中的数据路径使用**相对路径**（基准是项目根目录 `用户识别算法/`），
便于项目在他人电脑上直接使用：

- 原始数据已放在项目根目录下的 `data/` 文件夹：
  `<项目根目录>\data\睡姿数据`（同目录下含 readme.md 身高体重信息）
- 运行结果输出到项目根目录下的 `output/`。

若数据不在上述位置，两种改法任选其一：

1. 设置环境变量覆盖数据路径（推荐，不改代码）：
   ```bash
   # Windows
   set DATA_ROOT=你的数据目录
   # Linux / macOS
   export DATA_ROOT=/你的/数据/目录
   ```
2. 直接修改 `user_recognition/config.py` 中的 `DATA_ROOT`。

## 使用方法

```bash
pip install -r requirements.txt

# 完整流程（训练 + 评估 + 可视化 + 导出接口）
python main.py

# 复用已训练模型，仅重新可视化 / 导出（不重训）
python main.py --visualize-only
```

运行结束会在 `output/` 下生成：

- `embedder.pt` / `scaler.npz` —— 嵌入网络权重与标准化参数
- `embedding_tsne.png` —— 用户特征二维降维区分度图（未注册用户用黑圈标出）
- `score_dist.png` —— 已注册/未注册用户最高相似度分布
- `threshold_curve.png` —— FAR/FRR 随阈值变化曲线
- `recognition_registered.png` / `recognition_unknown.png` —— 两类识别示例
- 若干 `.json` —— 前端可用的数据接口（见下）

## 数据接口（供前端 / 其他成员使用）

所有接口文件为 UTF-8 JSON，位于 `output/`。

### 1. `users.json` —— 用户名单与身高体重

```json
{
  "api": "user_list",
  "count": 33,
  "users": [
    { "name": "dgs", "height": 178, "weight": 75 },
    ...
  ]
}
```

### 2. `templates.json` —— 已注册用户的模板库

```json
{
  "api": "user_templates",
  "embed_dim": 64,
  "count": 23,
  "users": [
    { "name": "ghx", "height": 172, "weight": 65,
      "template": [0.124, ...], "raw_count": 999 },
    ...
  ]
}
```

### 3. `recognition_result.json` —— 最近一次识别结果

```json
{
  "api": "recognition",
  "recognized_user": "dgs",       // 为空(null)表示判为未注册用户
  "score": 0.996,
  "threshold": 0.985,
  "is_registered": true,
  "similarity": { "ghx": 0.12, ..., "dgs": 0.996 }
}
```

### 4. `features.json` —— 各用户若干帧的嵌入（用于绘制区分度图）

```json
{
  "api": "user_embeddings",
  "dim": 64,
  "samples": [
    { "user": "SAI", "embedding": [...], "is_registered": true },
    ...
  ]
}
```

### 5. `model_state.json` —— 模型元信息（训练/测试用户划分、维度等）

## 关键评估结果

按题目协议（70% 用户训练注册，30% 用户作为新用户测试）：

| 指标 | 数值 | 说明 |
|------|------|------|
| 已注册用户识别准确率 | 100% | 在阈值之上仍能区分全部未拒识样本 |
| 假接受率 FAR | ≈3.9% | 在题目要求 1%~5% 区间内 |
| 假拒绝率 FRR | ≈0.9% | 远低于 5%，说明已注册用户几乎不被误拒 |
| 特征区分度 | t-SNE 同用户聚簇 | 未注册用户与注册用户分离清晰 |

## 复现说明

- 训练/测试用户划分使用固定随机种子（`config.RANDOM_SEED=42`），可复现。
- 数据增强：`feature_engineering.augment_frame`（噪声/平移/增益/遮挡）。
- 若要展示"新用户加入"流程：把某测试用户帧作为输入，`recognize` 返回
  `user=None` 且 `is_registered=false`；把该用户存入模板库后再识别即可命中。