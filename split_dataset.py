import os
import shutil
import random

def split_kaggle_dataset():
    """
    将kaggle_3m数据集分割为train/valid/test
    """
    root_dir = './dataset/kaggle_3m'
    train_dir = os.path.join(root_dir, 'train')
    valid_dir = os.path.join(root_dir, 'valid')
    test_dir = os.path.join(root_dir, 'test')

    # 创建目录
    for dir_path in [train_dir, valid_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有患者文件夹
    patient_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f)) and f.startswith('TCGA_')]

    # 随机打乱
    random.seed(42)
    random.shuffle(patient_folders)

    # 分割比例：70% train, 20% valid, 10% test
    n_total = len(patient_folders)
    n_train = int(0.7 * n_total)
    n_valid = int(0.2 * n_total)
    n_test = n_total - n_train - n_valid

    train_patients = patient_folders[:n_train]
    valid_patients = patient_folders[n_train:n_train+n_valid]
    test_patients = patient_folders[n_train+n_valid:]

    print(f"总患者数: {n_total}")
    print(f"训练集: {len(train_patients)}")
    print(f"验证集: {len(valid_patients)}")
    print(f"测试集: {len(test_patients)}")

    # 移动文件夹
    for patient in train_patients:
        shutil.move(os.path.join(root_dir, patient), os.path.join(train_dir, patient))

    for patient in valid_patients:
        shutil.move(os.path.join(root_dir, patient), os.path.join(valid_dir, patient))

    for patient in test_patients:
        shutil.move(os.path.join(root_dir, patient), os.path.join(test_dir, patient))

    print("数据集分割完成！")

if __name__ == '__main__':
    split_kaggle_dataset()