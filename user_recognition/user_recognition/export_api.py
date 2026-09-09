# -*- coding: utf-8 -*-
"""JSON / 文件数据接口（供前端或其他成员使用）

导出的内容包括：
- model_state.json        —— 模型元信息（输入维度、嵌入维度、scaler 参数）。
                             实际权重为 embedder.pt（PyTorch）与 scaler.npz。
- templates.json          —— 用户模板库（每个用户中心向量 + 元信息）。
- users.json              —— 用户名单与身高体重信息。
- recognition_result.json —— 前端可直接读取的最近一次识别结果。
- features.json           —— 可选的各用户某几帧嵌入（用于前端绘制降维区分度）。

前端只需读取 templates.json 与 recognition_result.json 即可展示
"数据库中已存在用户的识别效果"和"未在训练数据库中用户的识别效果"。
"""

import os
import json
import datetime
import numpy as np

from . import config


def _dump(obj, fname):
    """JSON 序列化辅助：保证 UTF-8 编码且不转义中文，缩进 2 空格便于人读。"""
    path = os.path.join(config.OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def export_users_json(user_info, fname="users.json"):
    """导出用户名单与身高体重。"""
    items = [{"name": u,
              "height": info.get("height"),
              "weight": info.get("weight")}
             for u, info in sorted(user_info.items())]
    payload = {
        "api": "user_list",
        "count": len(items),
        "users": items,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return _dump(payload, fname)


def export_templates_json(library, user_info, fname="templates.json"):
    """导出模板库。library 需提供 .users 与 .centers(或 .templates)。

    兼容两类模板库：
    - UserTemplateLibrary（单中心）  -> 每用户一个 template 字段
    - MultiTemplateLibrary（按睡姿） -> 每用户 templates 子列表
    """
    if hasattr(library, "centers"):
        # 单中心模板库
        entries = []
        for u in library.users:
            entries.append({
                "name": u,
                "height": user_info.get(u, {}).get("height"),
                "weight": user_info.get(u, {}).get("weight"),
                "template": [float(x) for x in library.centers[u]],
                "raw_count": library.raw_count.get(u, 0),
            })
    else:
        # 按睡姿多中心模板库
        entries = []
        for u in library.users:
            sub = []
            for pose, c in library.templates[u]:
                sub.append({"pose": pose, "template": [float(x) for x in c]})
            entries.append({
                "name": u,
                "height": user_info.get(u, {}).get("height"),
                "weight": user_info.get(u, {}).get("weight"),
                "templates": sub,
                "raw_count": library.raw_count.get(u, 0),
            })
    payload = {
        "api": "user_templates",
        "embed_dim": library.embed_dim,
        "count": len(entries),
        "users": entries,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return _dump(payload, fname)


def export_recognition_result(result, fname="recognition_result.json"):
    """导出最近一次识别结果。result 为 matcher.recognize 的返回。

    前端通过 recognized_user 是否为 null 判断是"已注册用户"还是"陌生人"。
    """
    payload = {
        "api": "recognition",
        "recognized_user": result.get("user"),          # null 表示拒识
        "score": result.get("score"),
        "threshold": result.get("threshold"),
        "is_registered": result.get("user") is not None,
        "similarity": result.get("all"),                # 与每位用户的相似度
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return _dump(payload, fname)


def export_features_json(features, labels, fname="features.json", limit=2000,
                         registered_set=None):
    """导出嵌入（限制数量），供前端绘制降维区分度图。

    registered_set: 已注册用户集合，用于标注每个样本是否属于数据库中已有用户。
    """
    features = np.asarray(features)
    # 样本过多时均匀抽样，避免 JSON 文件过大（前端也不需要全部点）
    if len(features) > limit:
        idx = np.linspace(0, len(features) - 1, limit).astype(int)
        features = features[idx]
        labels = [labels[i] for i in idx]
    payload = {
        "api": "user_embeddings",
        "dim": int(features.shape[1]) if features.ndim > 1 else 0,
        "samples": [
            {
                "user": labels[i],
                "embedding": [float(x) for x in features[i]],
                "is_registered": (
                    labels[i] in registered_set if registered_set is not None
                    else True
                ),
            }
            for i in range(len(features))
        ],
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return _dump(payload, fname)


def export_model_meta(net, scaler, train_users, test_users,
                      fname="model_state.json"):
    """导出模型元信息。方便前端 / 后续加载时知道输入输出维度、用户划分。"""
    payload = {
        "api": "model_meta",
        "input_dim": net.input_dim,
        "embed_dim": net.embed_dim,
        "train_users": sorted(train_users),
        "test_users": sorted(test_users),
        # 只抽样前 5 个维度的均值/方差（完整 scaler 在 scaler.npz 二进制文件中）
        "scaler_mean_first5": [float(x) for x in scaler.mean_[:5]],
        "scaler_scale_first5": [float(x) for x in scaler.scale_[:5]],
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return _dump(payload, fname)