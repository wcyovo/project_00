import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_all_data
from data_augmentation import augment_dataset

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("🔄 正在加载数据...")
print("=" * 60)

# 1. 加载原始数据
X, y = load_all_data("data/raw")

print(f"\n📊 原始样本数: {len(X)}")

# 2. 数据增强
X_aug, y_aug = augment_dataset(X, y, augment_factor=2)

# 3. 将标签转为数字编码
label_to_id = {label: idx for idx, label in enumerate(np.unique(y_aug))}
y_numeric = np.array([label_to_id[label] for label in y_aug])

print(f"\n📋 标签映射: {label_to_id}")

# 4. 数据展平
print("\n🔄 正在展平数据...")
X_flat = X_aug.reshape(X_aug.shape[0], -1)
print(f"📐 展平后特征维度: {X_flat.shape[1]}")

# 5. 划分训练集和测试集（70% 训练，30% 测试）
print("\n🔄 正在划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_numeric, test_size=0.3, random_state=42, stratify=y_numeric
)
print(f"📊 训练集: {len(X_train)} 样本")
print(f"📊 测试集: {len(X_test)} 样本")

# 6. 标准化数据
print("\n🔄 正在标准化数据...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. 训练SVM模型
print("\n" + "=" * 60)
print("🔄 正在训练 SVM 模型（使用增强数据）...")
print("=" * 60)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 8. 评估模型
print("\n📊 在测试集上评估...")
y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 详细分类报告:")
print(classification_report(y_test, y_pred, target_names=list(label_to_id.keys())))

# 9. 混淆矩阵
print("\n🔄 正在绘制混淆矩阵...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(label_to_id.keys()),
            yticklabels=list(label_to_id.keys()))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('SVM 混淆矩阵（增强数据）')
plt.tight_layout()
plt.savefig('models/svm_confusion_matrix_aug.png', dpi=150)
plt.show()

print("\n✅ SVM 模型训练完成（使用增强数据）！")
print("📁 混淆矩阵已保存至: models/svm_confusion_matrix_aug.png")