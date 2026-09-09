from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64
from project1.project_00.body_part.model import BodyPartRegressor

# ===== 解决中文乱码 =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# =======================

app = Flask(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model.pth'

model = BodyPartRegressor().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with open('data_2026.json', 'r', encoding='utf-8') as f:
    all_samples = json.load(f)

sleep_names = ['仰卧', '俯卧', '左侧卧', '右侧卧']

def predict(pressure, sleep_pose):
    pressure = np.array(pressure).reshape(44, 24)
    with torch.no_grad():
        p = torch.FloatTensor(pressure).unsqueeze(0).unsqueeze(0).to(DEVICE)
        pose = torch.LongTensor([sleep_pose]).to(DEVICE)
        pred = model(p, pose).squeeze().cpu().numpy()
    for i in range(0, 20, 2):
        pred[i] = pred[i] * 24
        pred[i+1] = pred[i+1] * 44
    boxes = []
    for i in range(5):
        boxes.append([float(pred[i*4]), float(pred[i*4+1]), float(pred[i*4+2]), float(pred[i*4+3])])
    return boxes

def parse_region(region_str):
    parts = region_str.split()
    x = [float(v) for v in parts[:10]]
    y = [float(v) for v in parts[12:22]]
    boxes = []
    for i in range(5):
        boxes.append([x[i*2], y[i*2], x[i*2+1], y[i*2+1]])
    return boxes

def generate_compare_image(pressure, true_boxes, pred_boxes, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 12))
    
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
    names = ['肩', '背', '腰', '臀', '大腿']
    
    # 左图
    im1 = ax1.imshow(pressure, cmap='jet', interpolation='bilinear', origin='upper')
    ax1.set_title('真实标注', fontsize=14)
    ax1.axis('off')
    if true_boxes:
        for i, box in enumerate(true_boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=colors[i], facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x1+2, y1+2, names[i], fontsize=10, color='white',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.7))
    
    # 右图
    im2 = ax2.imshow(pressure, cmap='jet', interpolation='bilinear', origin='upper')
    ax2.set_title('模型预测', fontsize=14)
    ax2.axis('off')
    if pred_boxes:
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=colors[i], facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x1+2, y1+2, names[i], fontsize=10, color='white',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.7))
    
    # ===== 颜色条放在最右侧 =====
    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('压力值', fontsize=12)
    
    plt.suptitle(title, fontsize=16, y=0.98)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/samples')
def get_samples():
    result = []
    for i, s in enumerate(all_samples):
        result.append({
            'index': i,
            'people_name': s['people_name'],
            'sleep_name': sleep_names[s['sleep_pos']],
            'action': s['action']
        })
    return jsonify(result[:200])

@app.route('/api/sample/<int:idx>')
def get_sample(idx):
    if idx >= len(all_samples):
        return jsonify({'error': 'index out of range'}), 400
    s = all_samples[idx]
    data_vals = [float(x) for x in s['data'].split(',')]
    return jsonify({
        'pressure': data_vals,
        'sleep_pose': s['sleep_pos'],
        'region': s['region'],
        'people_name': s['people_name'],
        'sleep_name': sleep_names[s['sleep_pos']],
        'action': s['action']
    })

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.json
    pressure = data['pressure']
    sleep_pose = data['sleep_pose']
    pred_boxes = predict(pressure, sleep_pose)
    
    true_boxes = None
    if 'region' in data:
        true_boxes = parse_region(data['region'])
    
    pressure_matrix = np.array(pressure).reshape(44, 24)
    title = f"{data.get('people_name', '')} | {sleep_names[sleep_pose]}"
    img_base64 = generate_compare_image(pressure_matrix, true_boxes, pred_boxes, title)
    
    return jsonify({
        'image': img_base64,
        'pred_boxes': pred_boxes,
        'true_boxes': true_boxes
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)