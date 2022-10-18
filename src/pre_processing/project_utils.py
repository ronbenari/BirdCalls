# exec(open("./project_utils.py").read())

import shutil
import os
import torch
# import dataloader
# from traintest import validate


def find_files_in_path(find_path, find_str):
    file_list = os.listdir(find_path)
    found = []
    for file_name in file_list:
        if find_str in file_name:
            found.append(file_name)
    print(f'Found {len(found)} files')
    return found


def move_files_from_to_folder(source_dir, target_dir, filter='WAV'):
    file_names = os.listdir(source_dir)

    for file_name in file_names:
        if filter is not None:
            if filter in file_name:
                shutil.move(os.path.join(source_dir, file_name), target_dir)
        else:
            shutil.move(os.path.join(source_dir, file_name), target_dir)

def label_str_tolist(label_str):
    label = label_str
    if ',' in label:
        label = label.split(',')
    if type(label) is not list:
        label = [label]
    return label

