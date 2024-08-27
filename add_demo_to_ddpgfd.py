'''
Descripttion: 
version: 
Author: He Guanhua
Date: 2024-08-27 16:30:03
LastEditors: He Guanhua
LastEditTime: 2024-08-27 16:30:48
LastEdition: 
'''
import re
import os
import shutil


def get_next_index(dst_dir):
    max_index = 0
    for file in os.listdir(dst_dir):
        match = re.match(r'demo_(\d+)\.pkl', file)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index + 1
    return max_index


def copy_and_rename_files(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    file_index = get_next_index(dst_dir)
    print(f"dst_dir had demo_{file_index - 1}")

    for subdir in os.listdir(src_dir):
        subdir_path = os.path.join(src_dir, subdir)
        print(f"curr dir is {subdir_path}")
        if os.path.isdir(subdir_path):
            demo_path = os.path.join(subdir_path, 'demo.pkl')
            dst_path = os.path.join(dst_dir, f'demo_{file_index}.pkl')
            shutil.copy(demo_path, dst_path)
            file_index += 1


src_dir = '/PATH/to/your/demos'
dst_dir = 'PATH/to/DDPGfD/data/demo'
copy_and_rename_files(src_dir, dst_dir)

