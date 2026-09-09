# -*- coding: utf-8 -*-
"""模板库与匹配识别

识别流程（匹配方式，符合"新用户加入尽量不修改模型"的要求）：
1. 注册（register）：将每个用户的多帧压入嵌入网络得到嵌入，
   计算该用户的"中心模板向量"（若干模板：全局中心或按睡姿分簇）。
2. 识别（recognize）：输入一帧（或多帧）-> 嵌入 -> 与全库模板计算
   余弦相似度；取最高相似度的用户作为候选。
3. 未知检测：若最高相似度低于阈值 tao，则判定为"未注册用户"（拒识），
   避免把新用户误认为某位已注册者。

相似度阈值的选取（FAR/FRR）在 evaluate.py 中完成。
"""

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


class UserTemplateLibrary:
    """每用户一个中心模板的模板库（主流程使用）。

    模板"中心" = 该用户所有注册帧嵌入的平均向量（再归一化）。
    用均值中心有两个好处：
    - 用多帧求平均，能抹掉单帧噪声，模板更稳定；
    - 只存一个向量/用户，注册新用户成本极低（O(1) 内存 + 一次均值）。
    """

    def __init__(self, embed_dim=64):
        self.embed_dim = embed_dim
        self.users = []            # 注册用户名单（注册顺序）
        self.centers = {}          # user -> 中心向量
        self.raw_count = {}        # user -> 注册帧数（信息量参考）

    def register_user(self, user, embeddings, mode="center"):
        """注册一个用户的模板。

        embeddings: (n_samples, embed_dim) 该用户若干帧的嵌入。
        mode: 'center' 全局均值中心。
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[None, :]
        center = emb.mean(axis=0)             # 沿帧维度求平均 = 中心模板
        cnorm = np.linalg.norm(center)
        if cnorm > 1e-8:
            center = center / cnorm           # 归一化，使余弦相似度可直接比较
        self.centers[user] = center.astype(np.float32)
        self.raw_count[user] = emb.shape[0]
        if user not in self.users:
            self.users.append(user)

    def has_user(self, user):
        """该用户是否已注册。"""
        return user in self.centers

    def template_matrix(self):
        """把全部中心向量堆叠成矩阵 (n_users, embed_dim)，便于批量计算。"""
        return np.stack([self.centers[u] for u in self.users])

    def recognize(self, query_embed, threshold=None):
        """匹配识别单条嵌入。

        返回 dict:
          {user: 匹配用户 or None(低于阈值), score, all: {user: sim}}
        """
        q = np.asarray(query_embed, dtype=np.float32).reshape(1, -1)
        T = self.template_matrix()
        # 一行 q 与所有模板做余弦相似度，得到 (1, n_users)
        sims = cosine_similarity(q, T)[0]
        best_idx = int(np.argmax(sims))        # 最大相似度的模板下标
        best_user = self.users[best_idx]
        best_score = float(sims[best_idx])
        if threshold is not None and best_score < threshold:
            # 最高分仍不够高 -> 判定为"未注册/陌生人"，返回 None
            best_user = None
        return {
            "user": best_user,
            "score": best_score,
            "threshold": threshold,
            "all": {u: float(s) for u, s in zip(self.users, sims)},
        }


class MultiTemplateLibrary:
    """每用户按睡姿建多个模板；识别时取最大相似度。

    mode='by_pose': 为每个用户注册 pose 0..3 的子中心，
    查询嵌入取与所有子模板的最高相似度。
    好处：同一用户不同睡姿的压力分布差异巨大，
    单中心可能无法覆盖所有睡姿；按睡姿多中心能提升姿态无关的匹配精度。
    """

    def __init__(self, embed_dim=64, n_poses=4):
        self.embed_dim = embed_dim
        self.n_poses = n_poses
        self.users = []
        self.templates = {}       # user -> list of (pose_id, center)
        self.raw_count = {}

    def register_user_by_pose(self, user, pose_embeds):
        """按睡姿注册。

        pose_embeds: {pose_id: (n, embed_dim) ndarray}
        对每个睡姿分别求中心，得到该用户在该睡姿下的子模板。
        """
        self.templates[user] = []
        total = 0
        for pose, emb in pose_embeds.items():
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb[None, :]
            if emb.shape[0] == 0:
                continue                       # 该睡姿无帧则跳过
            c = emb.mean(axis=0)
            n = np.linalg.norm(c)
            if n > 1e-8:
                c = c / n
            self.templates[user].append((int(pose), c.astype(np.float32)))
            total += emb.shape[0]
        self.raw_count[user] = total
        if user not in self.users:
            self.users.append(user)

    def recognize(self, query_embed, threshold=None):
        """多模板识别：对每个用户，取与该用户所有子模板的最高相似度，
        再在用户之间取全局最高。"""
        best_user = None
        best_score = -1.0
        all_scores = {}
        q = np.asarray(query_embed, dtype=np.float32).reshape(-1)
        qn = np.linalg.norm(q)
        for u in self.users:
            max_s = -1.0
            for _pose, center in self.templates[u]:
                # 等价于 cosine_similarity 的单点实现
                s = float(np.dot(q, center) / (qn * np.linalg.norm(center) + 1e-12))
                max_s = max(max_s, s)
            all_scores[u] = max_s
            if max_s > best_score:
                best_score = max_s
                best_user = u
        if threshold is not None and best_score < threshold:
            best_user = None
        return {
            "user": best_user,
            "score": best_score,
            "threshold": threshold,
            "all": all_scores,
        }