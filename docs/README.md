# SmartBed 项目文档

本目录存放团队的共享文档。各模块负责人请按约定在此更新文档，满足课程对"代码 + 文档"协作的要求。

## 文档索引

| 文档 | 说明 |
|---|---|
| [data-interface-protocol.md](data-interface-protocol.md) | **实时数据接口协议**：Unity 可视化端与算法/数据端的通信字段定义（团队对齐基线） |
| [mattress-spec.md](mattress-spec.md) | **床垫布局与数据规格**：压力点阵、睡姿类别、气囊分区、身体部位划分规格 |

## 团队成员分工（参考）

| 成员 | 负责内容 | 相关模块 |
|---|---|---|
| 组员A | ≥2 个睡姿识别算法 + 对比 | `src/`、`data/` |
| 组员B | 身体部位划分算法 | `body_part/` |
| 组员C | 弱力区域增强算法 | `src/`、`models/` |
| 你 | 基于实时数据的智能床垫可视化（Unity，Windows 部署） | `SmartBed_Visualization/` |

## 通用约定

- 分支策略：功能分支开发 → 合并到 `main`。
- 文档用 Markdown，中文正文。
- 实时数据统一遵循 [data-interface-protocol.md](data-interface-protocol.md) 的字段定义。
