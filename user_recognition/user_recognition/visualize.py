# -*- coding: utf-8 -*-
"""可视化模块

生成两类图：
1. 特征降维区分度图（t-SNE，失败时回退 PCA）：把注册用户与未注册用户的
   高维嵌入降到 2 维，直观观察不同用户的特征是否聚成彼此分离的簇；
   未注册用户用黑色空心圈强调。
2. FAR/FRR 与阈值选择曲线、相似度分布直方图。
3. 识别效果示意（某帧的压力热力图 + 与各用户模板的相似度条形图）。

matplotlib 中文字体问题：在 Windows 上优先使用 Microsoft YaHei / SimHei，
保证图中中文（标题、图例、坐标轴）能正常显示。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # 无界面后端：只保存图片，不弹窗
import matplotlib.pyplot as plt

from . import config


def _setup_font():
    """配置中文字体与负号显示。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False   # 负号不用 unicode，避免字体缺失乱码


def save_fig(fig, name):
    """保存图形到 output/ 目录并释放资源。"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_embedding_2d(emb, labels, title="用户特征二维降维区分度",
                      colors=None, name="embedding_tsne.png",
                      unknown_mask=None, annotate=True):
    """降维并绘图。

    :param emb: (n, embed_dim) 嵌入向量
    :param labels: 每个样本对应的用户名
    :param colors: 每用户颜色 dict（可选，不传则自动配色）
    :param unknown_mask: bool 数组，True 表示未注册用户样本（黑色空心圈强调）
    :param annotate: 是否在每个用户簇质心处标注用户名
    """
    _setup_font()
    n = len(emb)
    try:
        # t-SNE 非线性降维，类簇更容易被"拉开"
        from sklearn.manifold import TSNE
        z = TSNE(n_components=2, random_state=config.RANDOM_SEED,
                 perplexity=min(30, max(5, n - 1))).fit_transform(emb)
    except Exception:
        # 样本过少时 t-SNE 可能失败，回退到线性 PCA
        from sklearn.decomposition import PCA
        z = PCA(n_components=2, random_state=config.RANDOM_SEED).fit_transform(emb)
        title += " (PCA)"

    fig, ax = plt.subplots(figsize=(10, 8))
    unames = sorted(set(labels))
    # colors 为空时逐用户自动分配 tab20 颜色
    cmap = colors if colors else {u: None for u in unames}
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(unames), 2)))

    for i, u in enumerate(unames):
        # 未指定颜色的用户使用调色板中第 i 个颜色
        if cmap.get(u) is None:
            cmap[u] = palette[i % len(palette)]
        sel = np.where(np.array(labels) == u)[0]
        ax.scatter(z[sel, 0], z[sel, 1], color=cmap[u], s=26,
                   alpha=0.85, label=u, edgecolors="none")

    # 未注册用户样本叠加空心黑圈，一眼区分哪些是"陌生人"
    if unknown_mask is not None:
        unk = np.where(np.asarray(unknown_mask))[0]
        ax.scatter(z[unk, 0], z[unk, 1], facecolors="none",
                   edgecolors="black", s=90, linewidths=1.2, label="未注册用户")

    if annotate:
        # 在每类样本的质心位置写用户名，便于辨认簇归属
        for u in unames:
            sel = np.where(np.array(labels) == u)[0]
            if len(sel):
                cx, cy = np.mean(z[sel], axis=0)
                ax.annotate(u, (cx, cy), fontsize=7, alpha=0.9,
                            ha="center", va="center")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=4, loc="best")
    ax.grid(True, alpha=0.3)
    return save_fig(fig, name)


def plot_score_distribution(gen, imp, name="score_dist.png",
                            taos=None, best=None):
    """绘制已注册用户(本人)与未注册用户的最高相似度分布，并标注阈值。

    :param gen: 注册用户（本人）各查询帧的最高相似度
    :param imp: 未注册用户（陌生人）各查询帧的最高相似度
    :param best: pick_threshold 选出的工作点（含 threshold/FAR/FRR）
    """
    _setup_font()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(gen, bins=60, alpha=0.6, label="已注册用户(本人帧)", color="steelblue")
    ax.hist(imp, bins=60, alpha=0.5, label="未注册用户(新用户帧)", color="tomato")
    if taos is not None and best is not None:
        # 竖虚线标出工作点阈值，便于观察两类分布在阈值两侧的分离程度
        ax.axvline(best["threshold"], color="k", ls="--", lw=1.5,
                   label=f"阈值 {best['threshold']:.3f}\nFAR {best['FAR']*100:.1f}% FRR {best['FRR']*100:.1f}%")
    ax.set_xlabel("与最近模板的最高相似度")
    ax.set_ylabel("样本数")
    ax.set_title("已注册 / 未注册用户最高相似度分布")
    ax.legend(fontsize=9)
    return save_fig(fig, name)


def plot_threshold_curve(rows, name="threshold_curve.png", best=None):
    """绘制 FAR/FRR 随阈值变化曲线。

    观察两条曲线的交点与目标区间：
    - FAR（红点）随阈值升高而下降；
    - FRR（蓝方）随阈值升高而上升。
    """
    _setup_font()
    taos = [r["threshold"] for r in rows]
    far = [r["FAR"] for r in rows]
    frr = [r["FRR"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(taos, far, marker="o", ms=3, label="FAR", color="tomato")
    ax.plot(taos, frr, marker="s", ms=3, label="FRR", color="steelblue")
    if best is not None:
        ax.axvline(best["threshold"], color="green", ls="--", lw=1.5,
                   label=f"工作点阈值 {best['threshold']:.3f}")
        # 橙色/蓝色带标出题目要求的目标区间
        ax.axhspan(0.01, 0.05, alpha=0.15, color="orange", label="FAR 目标 1~5%")
        ax.axhspan(0.05, 0.10, alpha=0.12, color="blue", label="FRR 目标 5~10%")
    ax.set_xlabel("相似度阈值")
    ax.set_ylabel("比例")
    ax.set_title("FAR / FRR 随阈值变化")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return save_fig(fig, name)


def plot_recognition_example(frame, result, user_info, name="recognition_example.png"):
    """绘制某帧的热力图与识别结果。

    :param frame: (44, 24) 单帧压力图
    :param result: matcher.recognize 的返回（含 user/score/threshold/all）
    :param user_info: 身高体重信息（本函数当前仅作预留，可用于展示被测者信息）
    """
    _setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    # 左图：压力热力图（turbo 色带直观显示受力分布）
    ax0 = axes[0]
    im = ax0.imshow(frame, cmap="turbo", interpolation="bilinear",
                    extent=[0, frame.shape[1], 0, frame.shape[0]])
    ax0.grid(True, alpha=0.3, lw=0.3)
    ax0.set_title("输入压力帧")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    # 右图：该帧与各注册用户模板的相似度条形图
    ax1 = axes[1]
    if result.get("all"):
        us = list(result["all"].keys())
        sc = [result["all"][u] for u in us]
        order = np.argsort(sc)[::-1][:10][::-1]   # 只展示前 10 名，升序排列便于 barh
        us10 = [us[i] for i in order]
        sc10 = [sc[i] for i in order]
        ax1.barh(us10, sc10, color="steelblue")
        # 竖线标出阈值：得分超过阈值才接受，否则拒识
        ax1.axvline(result["threshold"] or -1, color="orange", ls="--",
                    label=f"阈值 {result['threshold']:.3f}" if result["threshold"] else None)
        ax1.set_xlabel("余弦相似度")
        ax1.set_title(f"识别结果: {result['user']} (得分 {result['score']:.3f})")
        ax1.legend(fontsize=8)
    plt.tight_layout()
    return save_fig(fig, name)