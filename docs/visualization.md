# 可视化系统说明（SmartBed_Visualization）

> 本模块基于 **Unity / 团结引擎（Tuanjie 1.10.3 / 2022.3.62t15）** 实现智能床垫实时数据可视化，可部署 Windows。
> 无第三方依赖，全部通过运行时构建 UI 与 3D 床垫。

## 1. 功能与系统架构映射

| 架构图模块 | 本系统实现 |
|---|---|
| **大屏可视化系统** | 顶部标题 + 全屏仪表盘（深色驾驶舱风格） |
| **床垫受压数据** | 44×24 压力热力图、3D 床垫形变、压力指标（最大/平均/接触面指数）、传感点压力曲线 |
| **用户睡姿** | 睡姿实时显示（仰卧/俯卧/左侧卧/右侧卧） |
| **气垫状态** | 8 个气垫状态列表、气垫区着色、气垫充气量曲线 |

## 2. 功能说明（对应用户要求）

- **展示气囊调节对人体的支撑效果**：依据压力分布**自推** 8 个气垫充气量（区域压力越高→该区气垫越放气减压，使"调节→压力变化"自洽）；以气垫区着色 + 3D 床垫形变直观呈现。
- **实时检测并显示睡眠状态**：订阅协议中的 `sleepPosture` 显示当前睡姿。
- **实时展示压力指标**：本地计算最大压力、平均压力、接触面指数（kPa）。
- **实时展示气垫状态与传感点压力变化**：滚动窗口（默认 90 帧）绘制 8 气垫充气量与 3 个传感点压力的动态曲线。

## 3. 实时数据流

```
床垫压力(44x24) ──> 数据端(模拟器/算法) ──HTTP GET /stream──> Unity 轮询 → 解析协议JSON → 可视化
```

- 数据协议见 [data-interface-protocol.md](data-interface-protocol.md)。
- 当前用 **数据模拟器** 回放仓库真实数据 `data/raw/SAI/SAI_1.txt`（仰卧，多帧）。
- 组员算法/真机若提供协议 `/stream` 端点，把 Unity 的 `streamUrl` 指向即可接入。

## 4. 运行步骤（本地演示）

1. **启动数据模拟器**（任一方式）：
   - 命令行：`python SmartBed_Visualization/Tools/Simulator/pressure_simulator.py --loop`
   - 或双击 `SmartBed_Visualization/Tools/Simulator/run_simulator.bat`（本机专用，不入库）
   - 成功标志：`[信息] 模拟器已启动：http://127.0.0.1:5001/stream`
2. 用 **团结引擎** 打开项目：`D:\Projects\SmartBed_Visualization\SmartBed_Visualization`
3. **按 Play**：`SmartBedBootstrap` 自动创建整套 UI 并轮询数据（无需手动搭场景）。

## 5. Windows 打包

- **菜单方式**：菜单栏 `Build → Windows Build`（需先打开项目）。
- **命令行方式**：
  ```
  "C:\Program Files\Tuanjie\Hub\Editor\2022.3.62t15\Editor\Tuanjie.exe" \
    -batchmode -quit -projectPath "D:\Projects\SmartBed_Visualization\SmartBed_Visualization" \
    -buildTarget Win64 -executeMethod BuildWindows.Build -logFile build.log
  ```
- **产物**：`SmartBed_Visualization/Build/Windows/SmartBedVisualization.exe`。

## 6. 目录说明（本模块）

```
SmartBed_Visualization/
  Assets/Scripts/
    Data/         协议模型、HTTP 数据源
    Core/         压力指标计算
    Visualization/ 热力图、部位框叠加、3D床垫、折线图、气垫策略、区域映射
    UI/           主控制器(自动构建 UI)、自启动引导
    Editor/       Windows 打包脚本
  Tools/Simulator/ 数据模拟器(纯标准库)
```
