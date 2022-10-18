
import os
import torch
import pandas as pd
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import numpy as np
import math


import torchaudio
from audiomentations import Compose, PitchShift, TimeStretch, Gain



class PreTrainAugmentFiles:
    def __init__(self, json_train_file, target_path, args_path=None, labels_path=None):
        self.json_train_file = json_train_file
        self.target_path = target_path

        self.args_path = args_path
        self.labels_path = labels_path

        with open(json_train_file, 'r') as fp:
            data_json = json.load(fp)

        self.train_data = data_json['data']

    # dst_file_name, json_dist_dir = 'C:/Learning/Deep Learning/project/labeled_data'
    def create_augmented_files(self, json_dst_dir):

        aug_file_data_list = [] 
        wav_dst_path = Path(self.target_path, 'train_aug')
        if not os.path.exists(wav_dst_path):
            os.mkdir(wav_dst_path)

        for train_data in self.train_data:
            train_file = train_data["wav"] 
            label_str  = train_data['labels']

            # make augmentatin to wav and save in new json both original and augmented
            aug_wav_file, sample_rate = apply_augmentation(train_file)

            train_file_path_p = Path(train_file)
            file_stem = train_file_path_p.stem 

            aug_file_stem = file_stem + '_aug'

            #aug_file_path = train_file.replace(file_stem, aug_file_stem) 
            #aug_file_path = aug_file_path.replace('train', 'train_aug')

            #aug_file_path = Path(wav_dst_path, aug_file_stem + '.WAV')
            wav_dst_path_str = str(Path(wav_dst_path))
            aug_file_path = wav_dst_path_str + '\\' + aug_file_stem + '.WAV'

            # saves the augmented file in new path
            torchaudio.save(aug_file_path, aug_wav_file, sample_rate) 

            aug_data = {
                'wav': aug_file_path,
                'labels': label_str
            }
            aug_file_data_list.append(aug_data)  

        # join 2 lists together
        file_data_list =  self.train_data + aug_file_data_list 

        dst_file_name =  'birdcalls_train_data_aug.json'

        # save new and existing files in json
        dst_file = Path(json_dst_dir, dst_file_name)
        with open(dst_file, 'w') as f:
            json.dump({'data': file_data_list}, f, indent=1)


pre_train_augment = Compose(
            [
                TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=1.0)
            ]
        )


def apply_augmentation(filename_path):
    """ makes augmentation to audio data
    receives pytorch tensor then numpy then returns back pytorch augmented waveform
    """
    waveform, sample_rate = torchaudio.load(filename_path)

    pre_train_augment.randomize_parameters(samples=waveform.numpy(), sample_rate=sample_rate)
    augmented_waveform = pre_train_augment(samples=waveform.numpy(), sample_rate=sample_rate)
    augmented_waveform_torch = torch.from_numpy(augmented_waveform)
    return augmented_waveform_torch, sample_rate