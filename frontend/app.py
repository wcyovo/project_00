import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 GUI 问题
from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import io
import base64
from collections import defaultdict

# 添加 src 到路径，以便导入 data_loader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_loader import load_all_data, load_pressure_file, parse_filename, num_to_label

app = Flask(__name__)
CORS(app)

# 设置中文字体（保存图片时使用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 全局数据缓存 ====================

# 缓存所有数据
all_data = None
all_labels = None
all_filenames = None
user_data = {}  # {用户名: [帧数据列表]}
user_labels = {}  # {用户名: [标签列表]}
user_frames_info = {}  # {用户名: {动作编号: [帧索引列表]}}
all_users = []

def load_and_cache_data():
    """加载并缓存所有数据"""
    global all_data, all_labels, all_filenames, user_data, user_labels, all_users, user_frames_info
    
    print("🔄 正在加载数据...")
    
    # 使用 data_loader 加载数据
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data/raw')
    
    # 收集所有文件
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt') and '动态' not in file and '空载' not in file:
                all_files.append(os.path.join(root, file))
    
    # ========== 添加自然排序（关键修复！）==========
    import re
    def natural_sort_key(filepath):
        """从文件路径中提取文件名，按数字自然排序"""
        filename = os.path.basename(filepath)
        return [int(part) if part.isdigit() else part 
                for part in re.split(r'(\d+)', filename)]
    
    all_files.sort(key=natural_sort_key)
    
    # 打印排序验证
    print("📋 文件排序验证（前10个）:")
    for i, f in enumerate(all_files[:10]):
        print(f"   {i+1}. {os.path.basename(f)}")
    # ========== 自然排序结束 ==========
    
    X = []
    y = []
    filenames = []
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        name, action_num = parse_filename(filename)
        
        if name is None:
            continue
        
        label = num_to_label.get(action_num)
        if label is None:
            continue
        
        frames = load_pressure_file(filepath)
        for frame in frames:
            if frame.shape == (44, 24):
                X.append(frame.tolist())
                y.append(label)
                filenames.append(filename)
    
    all_data = X
    all_labels = y
    all_filenames = filenames
    
    # 按用户分组
    user_data = defaultdict(list)
    user_labels = defaultdict(list)
    user_frames_info = defaultdict(lambda: defaultdict(list))
    
    for i, filename in enumerate(filenames):
        name, action_num = parse_filename(filename)
        user_data[name].append(X[i])
        user_labels[name].append(y[i])
        idx = len(user_data[name]) - 1
        user_frames_info[name][action_num].append(idx)
    
    all_users = list(user_data.keys())
    
    print(f"✅ 数据加载完成！")
    print(f"   总样本数: {len(X)}")
    print(f"   用户数: {len(all_users)}")
    print(f"   用户列表: {all_users[:10]}...")

# 启动时加载数据
load_and_cache_data()

# ==================== 路由 ====================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html', users=all_users)

@app.route('/api/users')
def get_users():
    """获取所有用户列表"""
    return jsonify({
        'users': all_users,
        'total_users': len(all_users)
    })

@app.route('/api/user/<username>')
def get_user_data(username):
    """获取指定用户的数据"""
    if username not in user_data:
        return jsonify({'error': '用户不存在'}), 404
    
    data = user_data[username]
    labels = user_labels[username]
    
    # 统计各睡姿数量
    posture_counts = {}
    for label in labels:
        posture_counts[label] = posture_counts.get(label, 0) + 1
    
    return jsonify({
        'username': username,
        'total_frames': len(data),
        'posture_counts': posture_counts,
        'postures': list(set(labels))
    })

@app.route('/api/user/<username>/heatmap')
def get_heatmap(username):
    """获取指定用户的热力图"""
    if username not in user_data:
        return jsonify({'error': '用户不存在'}), 404
    
    data = user_data[username]
    labels = user_labels[username]
    
    # 获取参数
    frame_idx = request.args.get('idx', 0, type=int)
    posture = request.args.get('posture', None)
    
    # 如果指定了睡姿，找到该睡姿的对应帧
    if posture:
        # 找到该睡姿的所有帧索引
        posture_indices = [i for i, label in enumerate(labels) if label == posture]
        if posture_indices:
            # 如果当前帧不在该睡姿中，跳转到该睡姿的第一帧
            if frame_idx not in posture_indices:
                frame_idx = posture_indices[0]
            # 否则保持当前帧
        else:
            frame_idx = 0
    else:
        # 没有指定睡姿，使用传入的 idx
        if frame_idx >= len(data):
            frame_idx = 0
    
    # 确保帧索引有效
    if frame_idx >= len(data):
        frame_idx = 0
    
    frame_data = np.array(data[frame_idx])
    label = labels[frame_idx]
    
    # 生成热力图
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(frame_data, cmap='turbo', interpolation='bilinear')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'{username} - 第{frame_idx}帧 - {label}', fontsize=12)
    
    # 转为 base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return jsonify({
        'username': username,
        'frame_idx': frame_idx,
        'posture': label,
        'image': f'data:image/png;base64,{image_base64}',
        'total_frames': len(data)
    })

@app.route('/api/user/<username>/frames')
def get_user_frames(username):
    """获取指定用户的所有帧信息"""
    if username not in user_data:
        return jsonify({'error': '用户不存在'}), 404
    
    data = user_data[username]
    labels = user_labels[username]
    
    frames_info = []
    for i, (frame, label) in enumerate(zip(data, labels)):
        frames_info.append({
            'idx': i,
            'posture': label,
            'shape': (len(frame), len(frame[0]))
        })
    
    # 按睡姿分组统计（合并所有相同睡姿的动作）
    posture_groups = {}
    for i, label in enumerate(labels):
        if label not in posture_groups:
            posture_groups[label] = {'start_idx': i, 'count': 0}
        posture_groups[label]['count'] += 1
    
    # 打印调试信息
    print(f"📊 用户 {username} 的睡姿分组（合并后）:")
    for posture, info in posture_groups.items():
        print(f"   {posture}: 起始索引 {info['start_idx']}, 数量 {info['count']}")
    
    return jsonify({
        'username': username,
        'total_frames': len(data),
        'frames': frames_info,
        'posture_groups': posture_groups
    })

@app.route('/api/statistics')
def get_statistics():
    """获取全局统计数据"""
    # 统计各睡姿总数
    posture_counts = {}
    for label in all_labels:
        posture_counts[label] = posture_counts.get(label, 0) + 1
    
    # 统计用户各睡姿
    user_posture_counts = {}
    for user in all_users:
        counts = {}
        for label in user_labels[user]:
            counts[label] = counts.get(label, 0) + 1
        user_posture_counts[user] = counts
    
    return jsonify({
        'total_frames': len(all_data),
        'total_users': len(all_users),
        'posture_counts': posture_counts,
        'user_posture_counts': user_posture_counts
    })

@app.route('/api/postures')
def get_postures():
    """获取所有睡姿类别"""
    postures = list(set(all_labels))
    return jsonify({'postures': postures})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)