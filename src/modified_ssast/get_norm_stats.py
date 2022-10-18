# -*- coding: utf-8 -*-
# @Time    : 8/4/21 4:30 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_norm_stats.py

# this is a sample code of how to get normalization stats for input spectrogram

import torch
import numpy as np

import dataloader

def get_norm_stats(train_data_path, class_labels_indices_path, batch_size=100):
    # set skip_norm as True only when you are computing the normalization stats
    audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 16, 'timem': 16, 'mixup': 0.0,
                  'skip_norm': True, 'mode': 'train', 'dataset': 'birdscall'}

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(train_data_path, label_csv=class_labels_indices_path,
                audio_conf=audio_conf), batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)
    mean = []
    std = []
    for i, (audio_input, labels) in enumerate(train_loader):
        cur_mean = torch.mean(audio_input)
        cur_std = torch.std(audio_input)
        mean.append(cur_mean)
        std.append(cur_std)
        print(cur_mean, cur_std)
    print(np.mean(mean), np.mean(std))
    return np.mean(mean), np.mean(std)