# -*- coding: utf-8 -*-
"""用户识别算法主流程

用法：
    python main.py                 # 完整训练 + 评估 + 可视化 + 导出
    python main.py --visualize-only  # 复用已有模型，只做可视化与导出

流程：
1. 加载数据
2. 70/30 按用户划分训练（注册）用户 / 测试（新）用户
3. 增强训练数据，训练特征嵌入网络
4. 为注册用户构建模板库
5. 评估：注册用户识别准确率 + FAR/FRR + 阈值选择
6. 可视化：嵌入降维区分度图、相似度分布、阈值曲线、识别示例
7. 导出 JSON 数据接口
"""

import argparse
import os
import json
import numpy as np

from user_recognition import (config, data_loader, feature_engineering as fe,
                              train_embedder, evaluate, visualize)
from user_recognition import export_api
from user_recognition.embedder import PressureEmbedder
from user_recognition.matcher import UserTemplateLibrary


def main(visualize_only=False, epochs=20):
    print("=" * 60)
    print("用户识别算法：特征提取 + 模板匹配")
    print("=" * 60, flush=True)

    # ---------- 1. 加载数据 ----------
    # user_actions: {用户: {动作号: (帧数,44,24) ndarray}}
    # user_info:    {用户: {height, weight}}
    user_actions, user_info = data_loader.load_all_users()
    users = sorted(user_actions.keys())
    print(f"加载用户数据：{len(users)} 人", flush=True)

    model_path = os.path.join(config.OUTPUT_DIR, "embedder.pt")
    scaler_path = os.path.join(config.OUTPUT_DIR, "scaler.npz")

    if visualize_only and os.path.exists(model_path):
        # ---------- 复用已有模型（跳过训练） ----------
        # 从 npz 恢复标准化参数，从 model_state.json 恢复划分名单与维度，
        # 从 embedder.pt 恢复网络权重。好处：改可视化/导出时无需重训。
        npz = np.load(scaler_path)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.mean_ = npz["mean"]
        scaler.scale_ = npz["scale"]
        meta_path = os.path.join(config.OUTPUT_DIR, "model_state.json")
        meta = json.load(open(meta_path, encoding="utf-8"))
        train_users = set(meta["train_users"])
        test_users = meta["test_users"]
        net = PressureEmbedder.from_file(
            model_path,
            input_dim=meta["input_dim"],
            embed_dim=meta["embed_dim"],
            n_classes=len(train_users),        # 分类头维度需与训练一致
        )
        print("已加载已有模型。", flush=True)
    else:
        # ---------- 2~3. 按用户划分并训练 ----------
        # train_and_save 内部完成：70/30 划分 -> 构造增强训练集 ->
        # 训练网络 -> 保存 embedder.pt / scaler.npz，返回划分名单。
        net, scaler, train_users, test_users = train_embedder.train_and_save(
            user_actions, users, model_path=model_path, scaler_path=scaler_path,
            epochs=epochs)
        export_api.export_model_meta(net, scaler, train_users, test_users)

    # ---------- 4. 构建模板库（注册用户） ----------
    # 每个注册用户用其全部动作的前部帧嵌入求均值中心，作为其模板。
    library = UserTemplateLibrary(embed_dim=config.EMBED_DIM)
    emb_of = evaluate.embed_users_for_templates(net, scaler, user_actions,
                                                train_users)
    for u in train_users:
        library.register_user(u, emb_of[u])
    print(f"模板库注册用户：{len(library.users)}", flush=True)

    # ---------- 5. 评估 ----------
    # evaluate_match：注册用户用留出帧查自己的识别准确率(gen)，
    # 未注册用户全部帧作冒用查询(imp)。
    ev = evaluate.evaluate_match(net, scaler, user_actions, train_users,
                                 test_users, library)
    print(f"注册用户识别准确率(不拒识)：{ev['acc']*100:.2f}%", flush=True)

    # 扫描阈值得到 (FAR, FRR)，再按"FAR 1~5% 内 FRR 最小"选出工作点
    rows = evaluate.thresholds_fr(ev["gen"], ev["imp"],
                                  np.linspace(0.30, 0.995, 140))
    best = evaluate.pick_threshold(rows)
    print("阈值扫描：", flush=True)
    print(f"  最优工作点 阈值={best['threshold']:.3f}  "
          f"FAR={best['FAR']*100:.2f}%  FRR={best['FRR']*100:.2f}%", flush=True)

    # ---------- 6. 可视化 ----------
    # 预分配每用户颜色（识别示例图复用）
    colors = {u: visualize.plt.cm.tab20(i / max(len(train_users), 1))
              for i, u in enumerate(sorted(train_users))}

    # 收集展示用的嵌入：注册用户 + 未注册用户各若干帧，
    # 用于 t-SNE 区分度图，并标注哪些属于未注册用户。
    emb_samples = []
    labels = []
    unknown = []
    emb_all = dict(emb_of)  # train 用户的嵌入（模板帧）
    for u in sorted(test_users):
        emb_all[u] = evaluate.embed_users_for_templates(
            net, scaler, user_actions, [u])[u]
    for u in sorted(train_users):
        # 每用户均匀抽 12 帧，避免点太多图太拥挤
        sel = np.linspace(0, len(emb_all[u]) - 1,
                          min(12, len(emb_all[u]))).astype(int)
        emb_samples.append(emb_all[u][sel])
        labels += [u] * len(sel)
        unknown += [False] * len(sel)
    for u in sorted(test_users):
        sel = np.linspace(0, len(emb_all[u]) - 1, 12).astype(int)
        emb_samples.append(emb_all[u][sel])
        labels += [u] * len(sel)
        unknown += [True] * len(sel)
    emb_samples = np.vstack(emb_samples)
    visualize.plot_embedding_2d(
        emb_samples, labels, colors=colors, unknown_mask=np.array(unknown),
        name="embedding_tsne.png",
        title="用户特征嵌入的二维降维区分度（● 未注册用户）")

    # 得分分布 + 阈值曲线
    visualize.plot_score_distribution(ev["gen"], ev["imp"],
                                      taos=[r["threshold"] for r in rows],
                                      best=best)
    visualize.plot_threshold_curve(rows, best=best)

    # 识别示例一：已注册用户（应命中本人，且得分高于阈值）
    demo_user = sorted(train_users)[0]
    demo_action = list(user_actions[demo_user].keys())[0]
    frame = user_actions[demo_user][demo_action][-1]    # 取最后一帧
    vec = fe.frame_to_vector(frame)
    q = net.predict_embed(scaler.transform(vec.reshape(1, -1)))[0]
    res = library.recognize(q, threshold=best["threshold"])
    visualize.plot_recognition_example(
        frame, res, user_info, name="recognition_registered.png")
    print(f"识别示例(注册用户 {demo_user}): "
          f"{res['user']} 得分 {res['score']:.3f}", flush=True)

    # 识别示例二：未注册用户（应被拒识 -> user 为 None）
    demo_u2 = sorted(test_users)[0]
    demo_a2 = list(user_actions[demo_u2].keys())[0]
    frame2 = user_actions[demo_u2][demo_a2][-1]
    vec2 = fe.frame_to_vector(frame2)
    q2 = net.predict_embed(scaler.transform(vec2.reshape(1, -1)))[0]
    res2 = library.recognize(q2, threshold=best["threshold"])
    visualize.plot_recognition_example(
        frame2, res2, user_info, name="recognition_unknown.png")
    print(f"识别示例(未注册用户 {demo_u2}): "
          f"{res2['user']}(None 表示正确拒识) 得分 {res2['score']:.3f}", flush=True)

    # ---------- 7. 导出 JSON 数据接口（供前端） ----------
    export_api.export_users_json(user_info)
    export_api.export_templates_json(library, user_info)
    # recognition_result.json 记录"最近一次识别"（即上面的陌生人示例），
    # 让前端可以直接渲染识别结果页面。
    export_api.export_recognition_result(res2)
    export_api.export_features_json(emb_samples, labels,
                                    registered_set=set(train_users))
    print("JSON 数据接口已导出到 output/ 目录。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用户识别算法")
    parser.add_argument("--visualize-only", action="store_true",
                        help="复用已有模型，仅重新可视化")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    main(visualize_only=args.visualize_only, epochs=args.epochs)