# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py.py

# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import csv
import json
import wave
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random

from audiomentations import Compose, AddGaussianNoise, TimeStretch, Shift, AirAbsorption
from audiomentations import Gain, GainTransition, AddGaussianSNR, TanhDistortion, ClippingDistortion
from audiomentations import Resample

from torch_audiomentations import PitchShift 
import math

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

augment = Compose([
#    AddGaussianSNR(min_snr_in_db=7, max_snr_in_db=10, p=0.5)
#    AddGaussianNoise(min_amplitude=0.015, max_amplitude=0.05, p=0.5)  #best but if only it is turend on
#    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)
#    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
#    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
#    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
#    AirAbsorption(min_temperature=10, max_temperature=20, p=0.5),

# new one 
    # PitchShift(min_semitones=-3, max_semitones=3, p=0.5) # long run
    # Shift(min_fraction=-0.5, max_fraction=0.5, p = 0.5)
    # Gain(min_gain_in_db = 5, max_gain_in_db = 15, p = 0.5),
    # GainTransition(min_gain_in_db = 5, max_gain_in_db = 15, p = 0.5)  # long run a bit
    # TanhDistortion(min_distortion=0.01, max_distortion=0.7, p = 0.5)
    # TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5), # long run (epoch 6.5 min)
    #ClippingDistortion(min_percentile_threshold=5, max_percentile_threshold=10, p=0.5)

    # Resample(min_sample_rate=182000, max_sample_rate= 202000, p=0.5) - long and not too good

# new Signal composition run 
    AddGaussianNoise(min_amplitude=0.015, max_amplitude=0.05, p=0.5),
    #AddGaussianSNR(min_snr_in_db=7, max_snr_in_db=10, p=0.5),
    ClippingDistortion(min_percentile_threshold=5, max_percentile_threshold=10, p=0.5),
    TanhDistortion(min_distortion=0.01, max_distortion=0.7, p = 0.5),
    Gain(min_gain_in_db = 5, max_gain_in_db = 15, p = 0.2), 
    Shift(min_fraction=-0.5, max_fraction=0.5, p = 0.5)
])

def WowRes(x):
    r = np.random.randint(2)
    if r == 1:
        res = x + 3* torch.div( torch.sin( 4* torch.tensor(math.pi)*x ), 4* torch.tensor(math.pi) )
    else:
        res = x    
    return res

def perf_audiomentations(waveform, sr):
    """ makes augmentation to audio data
    receives pytorch tensor then numpy then returns back pytorch augmented waveform
    """
    augmented_waveform = augment(samples=waveform.numpy(), sample_rate=sr)
    augmented_waveform_torch = torch.from_numpy(augmented_waveform)
    return augmented_waveform_torch


def perf_torch_audiomentations(waveform, sr):

    apply_augmentation = Compose(
        transforms=[
            #Gain(min_gain_in_db=-15.0, max_gain_in_db=5.0, p=0.5, ),
            #PolarityInversion(p=0.5)
            PitchShift(sample_rate=sr, min_transpose_semitones=-4, max_transpose_semitones=4, p = 0.5)
        ]
    )
    
    r_waveform = waveform[None, :]
    taugmented_waveform = apply_augmentation(samples=r_waveform, sample_rate=sr)
    taugmented_waveform = taugmented_waveform.squeeze(dim = 0)
    return taugmented_waveform
    

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))
        #pavel augmentation
        self.mode = self.audio_conf.get('mode')
        self.aug_en = self.audio_conf.get('aug_en')
        if (self.mode == 'train') and ( self.aug_en == True):
            print('now applying audiomentations augmentation')

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            # apply augmentation
            if (self.mode == 'train') and ( self.aug_en == True):
                #waveform = WowRes(waveform)
                waveform = perf_audiomentations(waveform, sr)
                #waveform = perf_torch_audiomentations(waveform, sr)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            if (self.mode == 'train') and ( self.aug_en == True):
                waveform1 = perf_audiomentations(waveform1, sr)
                waveform2 = perf_audiomentations(waveform2, sr)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0-mix_lambda)
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            fbank, mix_lambda = self._wav2fbank(datum['wav'])
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0

            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        # squeeze back
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices

    def __len__(self):
        return len(self.data)