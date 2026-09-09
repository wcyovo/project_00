import os
import numpy as np
import matplotlib.pyplot as plt
import re

# ==================== 映射关系 ====================

posture_map = {
    '仰卧': [1, 2, 3, 4, 5, 6],
    '俯卧': [7, 8, 9],
    '左侧卧': [10, 11, 12, 13, 14, 15],
    '右侧卧': [16, 17, 18, 19, 20, 21]
}

num_to_label = {}
for label, nums in posture_map.items():
    for num in nums:
        num_to_label[num] = label

# ==================== 解析文件名 ====================

def parse_filename(filename):
    """
    解析文件名，提取人名和动作编号
    例如：dgs_1.txt → ('dgs', 1)
          SAI_1_1.txt → ('SAI_1', 1)
    """
    name_part = filename.replace('.txt', '')
    parts = name_part.split('_')
    if len(parts) >= 3:
        # 格式：SAI_1_1.txt
        name = f"{parts[0]}_{parts[1]}"
        action_num = int(parts[2])
        return name, action_num
    elif len(parts) == 2:
        # 格式：dgs_1.txt
        name = parts[0]
        action_num = int(parts[1])
        return name, action_num
    return None, None

# ==================== 读取单个文件 ====================

def load_pressure_file(filepath):
    """
    读取一个txt文件，返回所有帧的列表
    每帧是一个44行×24列的numpy数组
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if not content:
        return []
    
    frames = []
    # 帧之间用空行分隔（两个换行符）
    frame_strs = content.split('\n\n')
    
    for frame_str in frame_strs:
        if not frame_str.strip():
            continue
        rows = frame_str.strip().split('\n')
        matrix = []
        for row in rows:
            if row.strip():
                # 按逗号分割，转为浮点数
                values = [float(x) for x in row.strip().split(',') if x.strip()]
                if values:
                    matrix.append(values)
        if matrix:
            # 确保矩阵是 44x24
            arr = np.array(matrix)
            if arr.shape == (44, 24):
                frames.append(arr)
            else:
                # 尝试重塑
                try:
                    arr = arr.reshape(44, 24)
                    frames.append(arr)
                except:
                    pass
    
    return frames

# ==================== 加载全部数据 ====================

def load_all_data(data_dir):
    """加载所有数据，递归遍历所有子文件夹"""
    all_files = []
    # 递归遍历 data_dir 下的所有子文件夹
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # 只处理 .txt 文件，忽略"动态"和"空载"文件
            if file.endswith('.txt') and '动态' not in file and '空载' not in file:
                all_files.append(os.path.join(root, file))
    
    # 按文件名自然排序（dgs_1, dgs_2, ..., dgs_10, dgs_11, ...）
    def natural_sort_key(filename):
        basename = os.path.basename(filename)
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split('([0-9]+)', basename)]
    
    all_files.sort(key=natural_sort_key)
    
    print(f"📁 共找到 {len(all_files)} 个数据文件")
    
    X = []  # 特征
    y = []  # 标签
    file_count = 0
    error_count = 0
    skip_count = 0
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        name, action_num = parse_filename(filename)
        
        if name is None:
            error_count += 1
            continue
        
        label = num_to_label.get(action_num)
        if label is None:
            error_count += 1
            continue
        
        frames = load_pressure_file(filepath)
        for frame in frames:
            # 确保每帧都是 44x24 的形状
            if frame.shape == (44, 24):
                X.append(frame)
                y.append(label)
            else:
                skip_count += 1
        
        file_count += 1
        if file_count % 50 == 0:
            print(f"已处理 {file_count} 个文件...")
    
    print(f"✅ 成功加载 {file_count} 个文件，共 {len(X)} 帧数据")
    if error_count > 0:
        print(f"⚠️ 有 {error_count} 个文件解析失败")
    if skip_count > 0:
        print(f"⚠️ 有 {skip_count} 帧数据形状异常，已跳过")
    
    return np.array(X), np.array(y)

# ==================== 可视化函数 ====================

def plot_heatmap(data, ax=None, title=""):
    """在指定ax上绘制单张热力图"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(data, cmap='turbo', interpolation='bilinear')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    return ax

# ==================== 主程序 ====================

if __name__ == "__main__":
    data_dir = "data/raw"
    
    if not os.path.exists(data_dir):
        print(f"❌ 文件夹 '{data_dir}' 不存在！")
        print("请先创建 data/raw 文件夹，并将数据集文件放入其中。")
    else:
        # 递归查找所有文件
        all_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.txt') and '动态' not in file and '空载' not in file:
                    all_files.append(os.path.join(root, file))
        
        print(f"📁 共找到 {len(all_files)} 个数据文件")
        
        if all_files:
            # 测试读取第一个文件
            test_file = all_files[0]
            filename = os.path.basename(test_file)
            name, action_num = parse_filename(filename)
            label = num_to_label.get(action_num, "未知")
            
            print(f"\n📄 测试文件: {filename}")
            print(f"📂 路径: {test_file}")
            print(f"👤 受试者: {name}")
            print(f"🔢 动作编号: {action_num}")
            print(f"🏷️ 睡姿类别: {label}")
            
            frames = load_pressure_file(test_file)
            print(f"📊 帧数: {len(frames)}")
            
            if frames:
                print(f"📐 每帧尺寸: {frames[0].shape}")
                
                # 显示前3帧
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for i in range(min(3, len(frames))):
                    plot_heatmap(frames[i], ax=axes[i], title=f"第{i}帧")
                plt.tight_layout()
                plt.show()
                
                print("\n✅ 数据读取成功！")
            else:
                print("❌ 未能读取到任何数据帧")
        else:
            print("❌ 没有找到任何 .txt 文件")