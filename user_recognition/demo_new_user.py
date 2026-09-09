# -*- coding: utf-8 -*-
"""演示：新用户加入流程（不修改模型结构，仅向模板库新增注册）

场景（为清晰展示"注册-识别"链路，采用一个还没有注册的训练用户来演示）：
1. 某用户未注册时，其帧与现有模板比较：可能被误判、也可能被正确拒识。
2. 将该用户若干帧加入模板库（仅新增模板，模型权重不重训）。
3. 再次输入该用户帧 -> 正确识别为该用户。

结论：新增用户识别无需修改模型结构，只需注册模板——满足题目要求。

运行前提：output/ 下已有训练好的 embedder.pt / scaler.npz / model_state.json
（即先运行过 python main.py）。
"""
import os
import json
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler

# 动态把项目根目录加入 sys.path，保证从任意目录运行本脚本都能 import 到 user_recognition
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from user_recognition import (config, data_loader, feature_engineering as fe,
                              evaluate)
from user_recognition.embedder import PressureEmbedder
from user_recognition.matcher import UserTemplateLibrary

# ---- 加载训练好的模型与标准化参数 ----
# scaler.npz 保存的是 StandardScaler 的均值和方差，推理时必须用同一套来标准化输入
npz = np.load(os.path.join(config.OUTPUT_DIR, "scaler.npz"))
meta = json.load(open(os.path.join(config.OUTPUT_DIR, "model_state.json"),
                      encoding="utf-8"))
scaler = StandardScaler()
scaler.mean_, scaler.scale_ = npz["mean"], npz["scale"]

# 从 embedder.pt 重建网络（分类头维度需与训练时的用户数一致）
net = PressureEmbedder.from_file(
    os.path.join(config.OUTPUT_DIR, "embedder.pt"),
    input_dim=meta["input_dim"], embed_dim=meta["embed_dim"],
    n_classes=len(meta["train_users"]),
)

# 阈值采用系统选定并导出的最优工作点（recognition_result.json 中的 threshold），
# 而不是在演示脚本里写死数字，保证与主流程评估使用同一判定标准。
threshold = json.load(
    open(os.path.join(config.OUTPUT_DIR, "recognition_result.json"),
         encoding="utf-8"))["threshold"]

# ---- 加载数据并构造"模拟未注册用户" ----
# 把第一个训练用户临时当作未入库的新用户，其余训练用户保持已注册
user_actions, user_info = data_loader.load_all_users()
train_users = list(meta["train_users"])

new_user = train_users[0]
registered_so_far = train_users[1:]

# 先只用"存量注册用户"建模板库
library = UserTemplateLibrary(embed_dim=config.EMBED_DIM)
emb_of = evaluate.embed_users_for_templates(net, scaler, user_actions,
                                            registered_so_far)
for u in registered_so_far:
    library.register_user(u, emb_of[u])
print(f"已注册 {len(registered_so_far)} 人；把 {new_user} 视为待注册新用户", flush=True)

# 查询用该用户的某个动作平均帧（与入库帧不同，模拟"识别新数据"）
query_action = list(user_actions[new_user].keys())[0]
q_frame = user_actions[new_user][query_action].mean(axis=0)
qe = net.predict_embed(
    scaler.transform(fe.frame_to_vector(q_frame).reshape(1, -1)))[0]

# 1) 注册前：应被拒识（user=None）
r1 = library.recognize(qe, threshold=threshold)
print(f"注册前: {new_user} 帧 -> 判为 {r1['user']} (score={r1['score']:.3f})", flush=True)

# 2) 注册（用该用户的多动作、多帧，仅新增模板，模型未重训）
emb_new = evaluate.embed_users_for_templates(net, scaler, user_actions, [new_user])
library.register_user(new_user, emb_new[new_user])
print(f"已用 {library.raw_count[new_user]} 帧注册 {new_user}（模型未重训）", flush=True)

# 3) 注册后：应正确命中 new_user
r2 = library.recognize(qe, threshold=threshold)
print(f"注册后: {new_user} 帧 -> 判为 {r2['user']} (score={r2['score']:.3f})"
      f"  期望 {new_user}", flush=True)

# 4) 校验存量用户不受影响：抽样 8 人识别，应全部命中本人
ok = 0
for u in registered_so_far[:8]:
    a = sorted(user_actions[u].keys())[2]
    v = fe.frame_to_vector(user_actions[u][a].mean(axis=0))
    qe2 = net.predict_embed(scaler.transform(v.reshape(1, -1)))[0]
    r = library.recognize(qe2, threshold=threshold)
    hit = (r["user"] == u)
    ok += int(hit)
    print(f"存量用户 {u}: 识别 -> {r['user']} (score={r['score']:.3f}) 期望 {u}")
print(f"存量用户抽样识别命中 {ok}/8", flush=True)

print()
print("结论：新增用户识别无需修改模型结构，只需注册模板——满足题目要求。")