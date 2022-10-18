
# exec(open("./data_validator.py").read())

# Validator
""""
1) file instance is once on file_list
2) each file is once on all train/validate/test jsons
3) files on train/validate/test directories match jsons
4) labels on jsons match excel (using labels_csv, and excel species columns)
"""
# from pydub import AudioSegment

import os
import pandas as pd
from pathlib import Path
import pickle
import json
# from tqdm import tqdm
import numpy as np
import math

from wav_processing import MarkedFiles, DatasetGenerator, data_split_str_list
from wav_processing import time_in_file_col, filename_col, recorder_col, unified_species_col
from wav_processing import no_call_label

from project_utils import label_str_tolist

pc_excel_path = '/media/inbalron/WinDrive/Ron Learning/Deep Learning/birds/Raw Records/audiomoth_general.xlsx'
pc_part_list_path = '/media/inbalron/WinDrive/Ron Learning/Deep Learning/birds/part_lists.json'
pc_labels_csv_path = '/media/inbalron/WinDrive/Ron Learning/Deep Learning/birds/Predictions/birdcalls_class_labels_indices.csv'
pc_labeled_path_dict = {
    'train': '/media/inbalron/WinDrive/Ron Learning/Deep Learning/birds/birdcalls_train_data.json',
    'validate': '/media/inbalron/WinDrive/Ron Learning/Deep Learning/birds/birdcalls_validate_data.json',
    'test': '/media/inbalron/WinDrive/Ron Learning/Deep Learning/birds/birdcalls_test_data.json'
    }


class Validator():
    """
    Validates the following
    1) file name is once on json file_list
    2) each file is once on all train/validate/test jsons together
    3) files on train/validate/test jsons match directories
    4) labels on jsons match excel (using labels_csv, and excel species columns)
    """
    def __init__(self, excel_path, labels_csv_path, part_list_path, labeled_path_dict):
        self.excel_path = excel_path
        self.labels_csv_path = labels_csv_path
        self.part_list_path = part_list_path
        self.labeled_path_dict = labeled_path_dict
        self.part_list_res = None
        self.labeled_jsons_res = None
        self.invalid_records = None
        # Load Data
        print('Loading excel from {excel_path}')
        try:
            self.mf = MarkedFiles(excel_path)
            self.mf.gen_unify_species_column()
        except:
            print('Error loading excel')
            return
        print('Loading label csv {labels_csv_path}')
        try:
            self.labels = pd.read_csv(labels_csv_path)
        except:
            print('Error loading labels csv')
            return
        print('Loading records part lists from {part_list_path}')
        try:
            with open(part_list_path, 'r') as f:
                self.part_list_dict = json.load(f)
        except:
            print('Error loading records part lists')
            return
        print('Loading train/validate/test jsons for SSAST run from {labeled_path_dict}')
        try:
            self.labeled_dict = {}
            for split_str in data_split_str_list:
                with open(labeled_path_dict[split_str], 'r') as f:
                    self.labeled_dict[split_str] = json.load(f)
        except:
            print('Error loading train/validate/test jsons')
            return
        print()
        # make label conversion dictionary from csv
        self.label_conversion_dict = dict(zip(self.labels['mid'].values, self.labels['display_name'].values))

    def check_part_list_occurrences(self):
        list_file_names = []
        non_unique_instances = []
        total_files_in_list_file = 0
        total_unique_files_per_three_parts = 0
        for part in self.part_list_dict.keys():
            total_files_in_list_file += len(self.part_list_dict[part])
            total_unique_files_per_three_parts += len(set(self.part_list_dict[part]))
            for file_path in self.part_list_dict[part]:
                if file_path in list_file_names:
                    non_unique_instances.append(file_path)
                else:
                    list_file_names.append(file_path)
        print(f'total_files_in_list_file {total_files_in_list_file}')
        print(f'total_unique_files_per_three_parts {total_unique_files_per_three_parts}')
        print(f'non_unique_instances {len(non_unique_instances)}')
        print('both totals should equal, non_unique_instances should be zero')
        print()
        res_dict = {
            'total_files_in_list_file': total_files_in_list_file,
            'total_unique_files_per_three_parts': total_unique_files_per_three_parts,
            'non_unique_instances': non_unique_instances
        }
        self.part_list_res = res_dict

    def check_labeled_jsons_occurrences(self):
        file_names = []
        non_unique_instances = []
        total_files_in_jsons = 0
        total_unique_files_per_three_parts = 0
        for part in self.labeled_path_dict.keys():
            file_list = []
            with open(self.labeled_path_dict[part], 'r') as f:
                labeled_dict_list = json.load(f)['data']
            for labeled_record in labeled_dict_list:
                file_list.append(labeled_record['wav'])
            total_files_in_jsons += len(file_list)
            total_unique_files_per_three_parts += len(set(file_list))
            for file_path in file_list:
                if file_path in file_names:
                    non_unique_instances.append(file_path)
                else:
                    file_names.append(file_path)
        print(f'total_files_in_jsons {total_files_in_jsons}')
        print(f'total_unique_files_per_three_parts {total_unique_files_per_three_parts}')
        print(f'non_unique_instances {len(non_unique_instances)}')
        print('both totals should equal, non_unique_instances should be zero')
        print()
        res_dict = {
            'total_files_in_jsons': total_files_in_jsons,
            'total_unique_files_per_three_parts': total_unique_files_per_three_parts,
            'non_unique_instances': non_unique_instances
        }
        self.labeled_jsons_res = res_dict

    def validate_labels_on_jsons(self, show=True):
        """
        for every record listed on train/validate/test jsons check that it matches excel marking
        use labels csv data to translate between json end excel
        :return:
        """
        self.invalid_json_records = []
        self.total_json_records = 0
        for part in self.labeled_path_dict.keys():
            file_list = []
            with open(self.labeled_path_dict[part], 'r') as f:
                labeled_dict_list = json.load(f)['data']
            for labeled_record_dict in labeled_dict_list:
                labeled_record = LabeledRecordItem(labeled_record_dict, self.label_conversion_dict)
                df = self.mf.marked_files
                record_df = df[(df[recorder_col] == labeled_record.metadata[recorder_col]) &
                               (df[filename_col] == labeled_record.metadata[filename_col])]
                valid_record = labeled_record.validate_record(record_df)
                if not valid_record:
                    self.invalid_json_records.append(labeled_record.record_full_path)
                    if show:
                        print(f'invalid record {labeled_record.record_full_path}')
                        print(f'record labels are {labeled_record.labels}')
                self.total_json_records += 1
        print()
        print(f'Found {len(self.invalid_json_records)} invalid records of {self.total_json_records} total records')
        print()


class LabeledRecordItem:
    def __init__(self, labeled_record_dict, label_conversion_dict=None):
        self.label_conversion_dict = label_conversion_dict
        self.record_full_path = labeled_record_dict['wav']
        self.record_name = Path(self.record_full_path).stem
        self.metadata = record_name_to_metadata(self.record_name)
        self.labels = labeled_record_dict['labels']
        inverse_conversion_dict = dict(zip(self.label_conversion_dict.values(), self.label_conversion_dict.keys()))
        self.no_call_label = inverse_conversion_dict[no_call_label]
        # Split labels if more than one, else save as list
        self.labels = label_str_tolist(self.labels)
        # if ',' in self.labels:
        #     self.labels = self.labels.split(',')
        # if type(self.labels) is not list:
        #     self.labels = [self.labels]

    def validate_record(self, record_df):
        record_valid = True
        mark_times = record_df[time_in_file_col].values
        # If label is no_call_label record_df should have no mark on that time
        # else record time should be on mark_times and records should match
        if self.labels == [self.no_call_label]:
            # print(f'record {self.record_name} is no call')
            if self.metadata[time_in_file_col] in mark_times:
                record_valid = False
        else:
            if self.metadata[time_in_file_col] not in mark_times:
                record_valid = False
            else:
                # check if label lists are same between record and marked excel
                mark_in_time = record_df[(record_df[time_in_file_col] == self.metadata[time_in_file_col])]
                marked_labels_list = mark_in_time[unified_species_col].values[0]
                labels_str_list = [self.label_conversion_dict[label] for label in self.labels]
                if len(labels_str_list) != len(marked_labels_list):
                    record_valid = False
                else:
                    for label in labels_str_list:
                        if label not in marked_labels_list:
                            record_valid = False
        return record_valid


def record_name_to_metadata(record_name):
    components = record_name.split('_')
    if len(components) == 4:
        record_metadata = {
            recorder_col: int(components[0]),
            filename_col: f'{components[1]}_{components[2]}',
            time_in_file_col: int(components[3])
        }
    else:
        print(f'Unknown record name "{record_name}"')
        record_metadata = {
            recorder_col: '',
            filename_col: '',
            time_in_file_col: ''
        }
    return record_metadata


# v = Validator(pc_excel_path, pc_labels_csv_path, pc_part_list_path, pc_labeled_path_dict)
dummy_labeled_record_dict = {'wav': '/dummy_dir/2_20200325_085700_20.wav', 'labels': '/m/birdc30'}

# df[(df[Gender]=='Male') & (df[Year]==2014)]

