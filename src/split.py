import os
import shutil
import random
import argparse
from get_data import get_data

def train_and_test(config_file):
    config = get_data(config_file)
    root_dir = config['data_source']['source_path']
    train_dir = config['data_source']['train_path']
    test_dir = config['data_source']['test_path']
    split_ratio = config['data_source']['split_ratio']  # e.g., 0.8

    # Create train and test directories
    for dir_path in [train_dir, test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Get class folders from source directory
    classes = os.listdir(root_dir)
    print(f"Found classes: {classes}")

    for class_name in classes:
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # Make subfolders for each class
        for t_dir in [os.path.join(train_dir, class_name), os.path.join(test_dir, class_name)]:
            os.makedirs(t_dir, exist_ok=True)

        # Copy files
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

    print("Data split complete ✅")

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml')
    passed_args = args_parser.parse_args()
    train_and_test(config_file=passed_args.config)
