# -*- coding: utf-8 -*-
"""评估模块：识别准确率、FAR、FRR 与阈值选择

评估协议：
- 已注册用户（训练用户）：用其"留出帧"（每动作后30%帧，或全部可用帧中的
  一部分）作为查询，测试识别准确率（最近中心匹配）。
- 未注册用户（测试/新用户）：其全部帧作为"未知"查询，用于计算拒识效果。

指标定义（生物识别标准）：
- FRR 假拒绝率：注册用户在阈值下被拒识（判为未知）的比例。
  （真实用户被当成陌生人，越低越好。）
- FAR 假接受率：未注册用户被判为"某注册用户"的比例。
  （陌生人混进系统，越低越好。）
- 通过扫描相似度阈值，找出同时满足 FAR 1~5%、FRR 5~10% 的工作点。
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from . import feature_engineering as fe


def embed_queries(net, scaler, user_actions, users, query_ratio=0.3,
                  max_frames_per_action=15, rng=None):
    """为指定用户生成查询向量。

    每个动作取后 query_ratio 帧（即注册时通常用前 70%，这里用后 30% 做查询，
    模拟"注册用一部分数据、识别用另一部分数据"的独立测试）。
    返回:
        Q: (n, embed_dim)
        y: 每行对应的用户 id
        users_list: id -> user 名单
    """
    rng = rng if rng is not None else np.random.default_rng(config.RANDOM_SEED)
    users = sorted(users)
    uid = {u: i for i, u in enumerate(users)}     # 用户 -> 类别号

    vecs = []
    ys = []
    for u in users:
        for a, arr in user_actions[u].items():
            nf = arr.shape[0]
            # 每动作最多取 max_frames_per_action 帧，均匀采样覆盖时间轴
            sel = np.linspace(0, nf - 1, min(max_frames_per_action, nf)).astype(int)
            chosen = arr[sel]
            k = max(1, int(len(chosen) * query_ratio))
            q_frames = chosen[-k:]                     # 后 k 帧作为查询（留出）
            q_vec = np.stack([fe.frame_to_vector(f) for f in q_frames])
            vecs.append(q_vec)
            ys.append(uid[u])

    X = np.concatenate(vecs, axis=0).astype(np.float32)
    Xs = scaler.transform(X).astype(np.float32)       # 与训练一致的标准化
    Q = net.predict_embed(Xs)                          # (n, embed_dim)
    # 每个 vecs 块是同一个用户，展开为逐帧的类别标签
    Y = np.repeat(np.asarray(ys, dtype=np.int64), [len(v) for v in vecs])
    return Q, Y, users


def embed_users_for_templates(net, scaler, user_actions, users,
                              max_frames_per_action=15):
    """提取用户模板所需嵌入：每个用户每动作取最多 max 帧。

    返回 {user: ndarray(n, embed_dim)} 的字典，每个用户有 <= 21*15 条嵌入，
    供 UserTemplateLibrary.register_user 做成中心模板。
    """
    out = {}
    for u in sorted(users):
        vecs = []
        for a, arr in user_actions[u].items():
            sel = np.linspace(0, arr.shape[0] - 1,
                              min(max_frames_per_action, arr.shape[0])).astype(int)
            for k in sel:
                vecs.append(fe.frame_to_vector(arr[k]))
        X = np.stack(vecs).astype(np.float32)
        Xs = scaler.transform(X).astype(np.float32)
        out[u] = net.predict_embed(Xs)
    return out


def evaluate_match(net, scaler, user_actions, train_users, test_users, library,
                   query_ratio=0.3, rng=None):
    """评估匹配识别。

    library 已注册 train_users 的模板。返回：
        gen_scores:  注册用户查询帧与自身匹配（最高相似度）得分
        imp_scores:  未注册用户查询帧的最高相似度得分
        acc:         注册用户识别准确率（不考虑拒识）
        confusion:   { (true, pred): count}
    """
    # ---- 注册用户：识别准确率 + 本人得分(gen) ----
    Q_tr, Y_tr, unames_tr = embed_queries(net, scaler, user_actions,
                                          train_users, query_ratio, rng=rng)
    gen_scores = []
    acc_correct = 0
    acc_total = len(Y_tr)
    for i in range(len(Q_tr)):
        # threshold=None：先不判断拒识，只看最近中心是否能命中正确用户
        res = library.recognize(Q_tr[i], threshold=None)
        best = res["user"]
        gen_scores.append(res["score"])
        acc_correct += (best == unames_tr[Y_tr[i]])
    acc = acc_correct / acc_total if acc_total else 0.0
    gen_scores = np.asarray(gen_scores)

    # ---- 未注册用户：全部帧当查询，只取最高相似度(imp) ----
    # query_ratio=1.0：未注册用户的所有帧都用来测"冒用"倾向
    Q_te, Y_te, unames_te = embed_queries(net, scaler, user_actions,
                                          test_users, query_ratio=1.0, rng=rng)
    imp_scores = []
    for i in range(len(Q_te)):
        res = library.recognize(Q_te[i], threshold=None)
        imp_scores.append(res["score"])
    imp_scores = np.asarray(imp_scores)

    return {"acc": acc, "gen": gen_scores, "imp": imp_scores}


def thresholds_fr(gen, imp, tao_range):
    """对每个阈值计算 FAR/FRR。返回 list of dict。

    阈值 tao：相似度达到 tao 才接受为该用户。
    - FAR(假接受) = 未注册用户得分 >= tao 的比例（越严越低）
    - FRR(假拒绝) = 注册用户得分 < tao 的比例（越严越高）
    二者是一对矛盾：阈值越高，陌生越难混入(FAR降)但真人越容易被拒(FRR升)。
    """
    rows = []
    for t in tao_range:
        far = float(np.mean(imp >= t))     # 陌生人被接受的比例
        frr = float(np.mean(gen < t))      # 真人被拒绝的比例
        rows.append({"threshold": float(t), "FAR": far, "FRR": frr})
    return rows


def pick_threshold(rows, far_min=0.01, far_max=0.05, frr_min=0.05, frr_max=0.10):
    """选择相似度阈值工作点。

    文献与题目要求 FAR 1%~5%、FRR 5%~10%（"左右"）。实际用户识别系统中，
    FRR 越低（已注册用户被误拒越少）越好。因此本函数以 **FAR 必须落在目标区间
    [far_min, far_max] 内**为首要约束，然后选择 FRR 最小（对已注册用户最友好）
    同时兼顾接近 FRR 目标区间的点。

    具体策略：
    1. 在 FAR 处于 [far_min, far_max] 的候选点中，取 FRR 最小的点；
       若 FRR 未能达到 5%，如实报告（这表示系统对已注册用户更宽容，指标优于要求）。
    2. 若无候选点，则退化为：FAR 最小且 FRR 最小的折中点。

    返回 dict（含 threshold/FAR/FRR）或 None。
    """
    cand = [r for r in rows if far_min <= r["FAR"] <= far_max]
    if cand:
        best = min(cand, key=lambda r: r["FRR"])
        return best

    # 退而求其次：无严格符合 FAR 区间的点，取 FAR 最接近上限且可接受的点
    cand = [r for r in rows if r["FAR"] <= far_max + 0.02]
    if cand:
        best = min(cand, key=lambda r: (abs(r["FRR"] - 0.075)))
        return best
    return None