import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter
from sklearn.model_selection import train_test_split
from data_loader import load_all_data

# ==================== 数据增强函数 ====================

def add_gaussian_noise(data, mean=0, std=0.01):
    """
    添加高斯噪声
    data: 输入数据 (44, 24) 或 (N, 44, 24)
    std: 噪声标准差（相对于数据范围的百分比）
    """
    if data.ndim == 2:
        noise = np.random.normal(mean, std * np.max(data), data.shape)
        return np.clip(data + noise, 0, None)  # 压力值不能为负
    else:
        result = data.copy()
        for i in range(len(data)):
            noise = np.random.normal(mean, std * np.max(data[i]), data[i].shape)
            result[i] = np.clip(data[i] + noise, 0, None)
        return result


def add_salt_pepper_noise(data, prob=0.01):
    """
    添加椒盐噪声（随机将部分像素置为0或最大值）
    prob: 噪声概率
    """
    if data.ndim == 2:
        result = data.copy()
        mask = np.random.random(data.shape) < prob
        result[mask] = np.max(data) * np.random.choice([0, 1], size=np.sum(mask))
        return result
    else:
        result = data.copy()
        for i in range(len(data)):
            mask = np.random.random(data[i].shape) < prob
            result[i][mask] = np.max(data[i]) * np.random.choice([0, 1], size=np.sum(mask))
        return result


def rotate_pressure_data(data, angle_range=(-5, 5)):
    """
    随机旋转压力数据（小角度旋转，保持人体形状）
    angle_range: 旋转角度范围（度）
    """
    if data.ndim == 2:
        angle = np.random.uniform(angle_range[0], angle_range[1])
        return rotate(data, angle, reshape=False, order=1, mode='reflect')
    else:
        result = data.copy()
        for i in range(len(data)):
            angle = np.random.uniform(angle_range[0], angle_range[1])
            result[i] = rotate(data[i], angle, reshape=False, order=1, mode='reflect')
        return result


def random_shift(data, shift_range=(-2, 2)):
    """
    随机平移压力数据（模拟人体在床上的轻微位移）
    shift_range: 平移范围（像素）
    """
    if data.ndim == 2:
        row_shift = np.random.randint(shift_range[0], shift_range[1] + 1)
        col_shift = np.random.randint(shift_range[0], shift_range[1] + 1)
        return np.roll(data, (row_shift, col_shift), axis=(0, 1))
    else:
        result = data.copy()
        for i in range(len(data)):
            row_shift = np.random.randint(shift_range[0], shift_range[1] + 1)
            col_shift = np.random.randint(shift_range[0], shift_range[1] + 1)
            result[i] = np.roll(data[i], (row_shift, col_shift), axis=(0, 1))
        return result


def apply_gaussian_blur(data, sigma=0.5):
    """
    高斯模糊（模拟传感器噪声平滑）
    sigma: 高斯核标准差
    """
    if data.ndim == 2:
        return gaussian_filter(data, sigma=sigma)
    else:
        result = data.copy()
        for i in range(len(data)):
            result[i] = gaussian_filter(data[i], sigma=sigma)
        return result


def random_scaling(data, scale_range=(0.95, 1.05)):
    """
    随机缩放压力值（模拟不同体重的压力差异）
    scale_range: 缩放范围
    """
    if data.ndim == 2:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return data * scale
    else:
        result = data.copy()
        for i in range(len(data)):
            scale = np.random.uniform(scale_range[0], scale_range[1])
            result[i] = data[i] * scale
        return result


# ==================== 复合数据增强函数 ====================

def augment_dataset(X, y, augment_factor=2):
    """
    对数据集进行数据增强
    X: 原始特征 (N, 44, 24)
    y: 原始标签
    augment_factor: 每个样本生成的增强样本数量（如2表示原样本+2个增强样本）
    
    返回增强后的数据集
    """
    print(f"\n🔄 正在执行数据增强...")
    print(f"   原始样本数: {len(X)}")
    
    X_aug = []
    y_aug = []
    
    # 保留原始数据
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])
    
    # 生成增强数据
    for i in range(len(X)):
        for _ in range(augment_factor):
            aug_data = X[i].copy()
            
            # 随机选择增强方法组合
            aug_methods = np.random.choice([
                'noise', 'salt_pepper', 'rotate', 'shift', 'blur', 'scale'
            ], size=np.random.randint(1, 3), replace=False)
            
            for method in aug_methods:
                if method == 'noise':
                    aug_data = add_gaussian_noise(aug_data, std=np.random.uniform(0.005, 0.02))
                elif method == 'salt_pepper':
                    aug_data = add_salt_pepper_noise(aug_data, prob=np.random.uniform(0.005, 0.015))
                elif method == 'rotate':
                    aug_data = rotate_pressure_data(aug_data, angle_range=(-3, 3))
                elif method == 'shift':
                    aug_data = random_shift(aug_data, shift_range=(-1, 1))
                elif method == 'blur':
                    aug_data = apply_gaussian_blur(aug_data, sigma=np.random.uniform(0.3, 0.7))
                elif method == 'scale':
                    aug_data = random_scaling(aug_data, scale_range=(0.97, 1.03))
            
            X_aug.append(aug_data)
            y_aug.append(y[i])
    
    print(f"   增强后样本数: {len(X_aug)}")
    print(f"   增强比例: {len(X_aug)/len(X):.2f}x")
    
    return np.array(X_aug), np.array(y_aug)


# ==================== 按用户划分数据集 ====================

def split_by_user(X, y, filenames, train_ratio=0.7):
    """
    按用户划分训练集和测试集（70% 用户训练，30% 用户测试）
    注意：此函数需要 filenames 参数，从 data_loader 中获取文件名列表
    """
    # 获取所有唯一用户
    unique_users = list(set([parse_user_from_filename(f) for f in filenames]))
    np.random.shuffle(unique_users)
    
    split_idx = int(len(unique_users) * train_ratio)
    train_users = set(unique_users[:split_idx])
    test_users = set(unique_users[split_idx:])
    
    # 按用户划分数据
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for i, filename in enumerate(filenames):
        user = parse_user_from_filename(filename)
        if user in train_users:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), train_users, test_users


def parse_user_from_filename(filename):
    """
    从文件名中提取用户ID
    例如：dgs_1.txt → 'dgs'
          SAI_1_1.txt → 'SAI_1'
    """
    name_part = filename.replace('.txt', '')
    parts = name_part.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}"
    elif len(parts) == 2:
        return parts[0]
    return name_part


# ==================== 可视化对比 ====================

def visualize_augmentation(X_original, y_original, X_aug, y_aug, sample_idx=0):
    """
    可视化原始数据和增强后的数据对比
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 原始数据
    axes[0, 0].imshow(X_original[sample_idx], cmap='turbo', interpolation='bilinear')
    axes[0, 0].set_title(f'原始数据\n标签: {y_original[sample_idx]}')
    axes[0, 0].axis('off')
    
    # 找到同一原始样本对应的增强样本
    # 由于增强样本是按顺序生成的，需要计算偏移
    # 假设 augment_factor=2，则增强样本在 X_aug 中的位置是：
    # 原样本: 0, 3, 6, ...; 增强1: 1, 4, 7, ...; 增强2: 2, 5, 8, ...
    
    # 简单方法：取前几个增强样本展示
    aug_indices = [sample_idx * 3 + 1, sample_idx * 3 + 2, sample_idx * 3 + 3]
    
    for i, idx in enumerate(aug_indices[:3]):
        if idx < len(X_aug):
            axes[0, i+1].imshow(X_aug[idx], cmap='turbo', interpolation='bilinear')
            axes[0, i+1].set_title(f'增强样本 {i+1}\n标签: {y_aug[idx]}')
            axes[0, i+1].axis('off')
    
    # 如果还有其他样本，显示更多
    if len(X_aug) > 4:
        axes[1, 0].imshow(X_aug[4], cmap='turbo', interpolation='bilinear')
        axes[1, 0].set_title('更多增强样本')
        axes[1, 0].axis('off')
    
    # 隐藏多余的子图
    for j in range(1, 4):
        axes[1, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('models/augmentation_visualization.png', dpi=150)
    plt.show()
    print("📁 增强可视化已保存至: models/augmentation_visualization.png")


# ==================== 主程序（测试数据增强效果） ====================

if __name__ == "__main__":
    # 加载数据
    print("=" * 60)
    print("🔄 正在加载原始数据...")
    print("=" * 60)
    
    X, y = load_all_data("data/raw")
    
    print(f"\n📊 原始样本数: {len(X)}")
    
    # 对数据进行增强
    X_aug, y_aug = augment_dataset(X, y, augment_factor=2)
    
    # 可视化增强效果
    visualize_augmentation(X, y, X_aug, y_aug, sample_idx=0)
    
    # 保存增强后的数据（可选）
    # np.save('data/processed/X_aug.npy', X_aug)
    # np.save('data/processed/y_aug.npy', y_aug)
    # print("\n✅ 增强数据已保存至 data/processed/")
    
    print("\n✅ 数据增强测试完成！")