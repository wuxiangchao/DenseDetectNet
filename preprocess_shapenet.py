# -*- coding: utf-8 -*-
# @Author: WUXIANGCHAO
# @Date: 2025-07-13 22:15:30
# @Last Modified by: WUXIANGCHAO
# @Last Modified time: 2025-08-22 10:30:00

import os
import numpy as np
import trimesh
from tqdm import tqdm
import multiprocessing

# 配置
SHAPENET_ROOT = './data/ShapeNetCore.v2' # ShapeNet根目录
CAR_CATEGORY_ID = '02958343'
NUM_POINTS = 2048 # 与模型定义保持一致
OUTPUT_DIR = os.path.join(SHAPENET_ROOT, 'processed_pointclouds')

def process_file(obj_path):
    """
    处理单个.obj文件：加载、采样并保存为.npy文件。
    """
    try:
        model_id = obj_path.split('/')[-3]
        # --------------------

        output_subdir = os.path.join(OUTPUT_DIR, CAR_CATEGORY_ID, model_id)
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, 'pointcloud.npy')

        # 如果文件已存在，则跳过
        if os.path.exists(output_path):
            return f"Skipped: {model_id}"

        # 使用 trimesh 加载和采样
        mesh = trimesh.load(obj_path, force='mesh')
        points, _ = trimesh.sample.sample_surface(mesh, NUM_POINTS)
        points = points.astype(np.float32)
        
        # 保存为.npy文件
        np.save(output_path, points)
        return f"Processed: {model_id}"
    except Exception as e:
        return f"Failed to process {obj_path}: {e}"

def main():
    print("--- Starting ShapeNet Pre-processing ---")
    
    # 查找所有.obj文件
    obj_paths = []
    car_dir = os.path.join(SHAPENET_ROOT, CAR_CATEGORY_ID)
    for model_id in os.listdir(car_dir):
        # 确保 model_id 是一个目录
        model_dir_path = os.path.join(car_dir, model_id)
        if os.path.isdir(model_dir_path):
            model_path = os.path.join(model_dir_path, 'models', 'model_normalized.obj')
            if os.path.exists(model_path):
                obj_paths.append(model_path)

    print(f"Found {len(obj_paths)} car models to process.")

    # 使用多进程并行处理以加快速度
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, obj_paths), total=len(obj_paths)))

    print("--- Pre-processing Finished ---")
    print(f"Processed point clouds are saved in: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
