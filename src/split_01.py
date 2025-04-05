from logging import root
import os
import shutil
import argparse
import yaml
import pandas as pd
import numpy as np
import random
from get_data import get_data, read_params
 
 
def train_and_test(config_file):
    config = get_data(config_file)
    root_dir = config['raw_data']['data_src']
    dest = config['load_data']['preproseesd_data']
   
    # Create destination directories if they don't exist
    os.makedirs(os.path.join(dest, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'test'), exist_ok=True)
   
    # plant disease dataset classes
    classes = [
        'Apple___Apple_scab',
        'Apple___Black_rot',    
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Cherry___Powdery_mildew',
        'Cherry___healthy',
        'Corn___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn___Common_rust_',
        'Corn___Northern_Leaf_Blight',
        'Corn___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
   
    # Create class directories in train and test
    for class_name in classes:
        os.makedirs(os.path.join(dest, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(dest, 'test', class_name), exist_ok=True)
   
    # Copy files from Training directory
    training_dir = os.path.join(root_dir, 'Training')
    for class_name in classes:
        src_dir = os.path.join(training_dir, class_name)
        if not os.path.exists(src_dir):
            print(f"Warning: Directory {src_dir} does not exist. Skipping...")
            continue
           
        files = os.listdir(src_dir)
        print(f"{class_name} (Training) -> {len(files)} images")
       
        for f in files:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dest, 'train', class_name, f)
            shutil.copy(src_path, dst_path)
           
        print(f"Done copying training data for {class_name}")
   
    # Copy files from Testing directory
    testing_dir = os.path.join(root_dir, 'Testing')
    for class_name in classes:
        src_dir = os.path.join(testing_dir, class_name)
        if not os.path.exists(src_dir):
            print(f"Warning: Directory {src_dir} does not exist. Skipping...")
            continue
           
        files = os.listdir(src_dir)
        print(f"{class_name} (Testing) -> {len(files)} images")
       
        for f in files:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dest, 'test', class_name, f)
            shutil.copy(src_path, dst_path)
           
        print(f"Done copying testing data for {class_name}")
 
 
if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    train_and_test(config_file=passed_args.config)