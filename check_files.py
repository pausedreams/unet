import os
from PIL import Image

def check_files():
    dirs = ['./dataset/kaggle_3m/train', './dataset/kaggle_3m/valid', './dataset/kaggle_3m/test']
    bad_files = []
    
    for data_dir in dirs:
        if not os.path.exists(data_dir):
            continue
        for patient in os.listdir(data_dir):
            patient_path = os.path.join(data_dir, patient)
            if os.path.isdir(patient_path):
                for file in os.listdir(patient_path):
                    if file.endswith('.tif') and not file.endswith('_mask.tif'):
                        image_path = os.path.join(patient_path, file)
                        mask_path = os.path.join(patient_path, file.replace('.tif', '_mask.tif'))
                        
                        try:
                            Image.open(image_path)
                            Image.open(mask_path)
                        except Exception as e:
                            bad_files.append((image_path, mask_path, str(e)))
    
    print(f"Found {len(bad_files)} bad file pairs")
    for img, mask, err in bad_files[:10]:  # 显示前10个
        print(f"Bad: {img}, {mask} - {err}")

if __name__ == '__main__':
    check_files()