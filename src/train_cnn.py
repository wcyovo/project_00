import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_all_data

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("🔄 正在加载数据...")
print("=" * 60)

# 1. 加载数据
X, y = load_all_data("data/raw")

print(f"\n📊 总样本数: {len(X)}")
print(f"📐 每帧尺寸: {X[0].shape}")
print(f"🏷️ 标签类别: {np.unique(y)}")

# 2. 将标签转为数字编码
label_to_id = {label: idx for idx, label in enumerate(np.unique(y))}
id_to_label = {idx: label for label, idx in label_to_id.items()}
y_numeric = np.array([label_to_id[label] for label in y])

print(f"\n📋 标签映射: {label_to_id}")

# 3. 划分训练集和测试集（70% 训练，30% 测试）
print("\n🔄 正在划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_numeric, test_size=0.3, random_state=42, stratify=y_numeric
)
print(f"📊 训练集: {len(X_train)} 样本")
print(f"📊 测试集: {len(X_test)} 样本")

# 4. 标准化数据（保留原始形状）
print("\n🔄 正在标准化数据...")
# 对每个像素位置独立标准化
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

# 恢复形状
X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)

print(f"📐 标准化后每帧尺寸: {X_train_scaled[0].shape}")

# 5. 使用 CNN（简单版本，用 sklearn 的 MLP 作为替代）
# 注意：由于环境限制，先使用 MLP（多层感知机）作为深度学习的替代
# 如果后续装了 torch，可以替换为真正的 CNN
print("\n" + "=" * 60)
print("🔄 正在训练 多层感知机 (MLP) 作为深度学习基准...")
print("=" * 60)

from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=50,
    random_state=42,
    verbose=True
)

# 再次展平用于 MLP
X_train_mlp = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
X_test_mlp = X_test_scaled.reshape(X_test_scaled.shape[0], -1)

mlp_model.fit(X_train_mlp, y_train)

# 6. 评估模型
print("\n📊 在测试集上评估...")
y_pred = mlp_model.predict(X_test_mlp)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 详细分类报告:")
print(classification_report(y_test, y_pred, target_names=list(label_to_id.keys())))

# 7. 混淆矩阵可视化
print("\n🔄 正在绘制混淆矩阵...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=list(label_to_id.keys()),
            yticklabels=list(label_to_id.keys()))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('MLP 混淆矩阵')
plt.tight_layout()
plt.savefig('models/mlp_confusion_matrix.png', dpi=150)
plt.show()

print("\n✅ MLP 模型训练完成！")
print("📁 混淆矩阵已保存至: models/mlp_confusion_matrix.png")
print("\n📌 提示: 如需真正的 CNN，请安装 torch 后使用 train_cnn_torch.py")