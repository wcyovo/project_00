# -*- coding: utf-8 -*-
"""用户识别算法包

完整流程：
1. data_loader    —— 读取并解析睡姿压力原始数据
2. embedder       —— 特征嵌入网络（softmax 训练的特征提取器）
3. matcher        —— 模板库构建、匹配识别、未知用户检测
4. train_embedder —— 训练特征提取器
5. evaluate       —— 准确率 / FAR / FRR 评估
6. visualize      —— 特征降维区分度图、识别效果热力图
7. export_api     —— 导出 JSON 数据接口（供前端使用）
"""