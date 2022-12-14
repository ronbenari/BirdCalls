
# exec(open("./wav_processing.py").read())


# from pydub import AudioSegment

import os
import torch
import pandas as pd
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import numpy as np
import math

from pydub import AudioSegment
from sklearn.model_selection import train_test_split

from project_utils import label_str_tolist

# working_path = '/content/gdrive/My Drive/DeepLearning'
# excel_path = '/content/gdrive/My Drive/DeepLearning/bird_records/audiomoth_general.xlsx'

# marked_files.columns
use_all_species_columns = True
if use_all_species_columns:
    species_columns = ['species 1', 'species 2', 'species 3', 'species 4']
else:
    species_columns = ['species 1'] # Use only first specie column
time_in_file_col = 'Time in file (sec)'
unified_species_col = 'unified species'
confirmed_col = 'Confirmed'
directory_col = 'Directory'
filename_col = 'File name'
recorder_col = 'Audiomoth number'
location_col = 'Audiomoth Location'

# class labels prefix
label_prefix = '/m/birdc'
labels_csv_columns = ['index', 'mid', 'display_name']
data_split_str_list = ['train', 'validate', 'test']
file_length_seconds = 30

no_call_label = 'NoCall'


class MarkedFiles:
    """
    Class for processing excel file with records classification
    Each row is a second on a record file
    Uses pandas
    """
    def __init__(self, excel_path):
        self.marked_files = pd.read_excel(excel_path, sheet_name='Classified species (birds)', index_col=None)
        self.label_info = None
        self.species = None
        self.species_str = None
        self.labels_dict = None
        self.labels_df = None
        self.labels_count = None
        self.label_mask = None
        self.confirmed_list = None
        self.json_black_list = None

    def get_directories(self):
        return self.marked_files[directory_col].unique()

    def get_file_list(self, base_path='', as_path=True):
        """
        generate records file list base on excel and the base directory they are saved in
        :param base_path: path where records are saved
        :param as_path: Use Path method from pathlib
        :return: list of all files on excel with path base path
        """
        file_list = []
        for k in range(len(self.marked_files)):
            row = self.marked_files.iloc[k]
            directory = row[directory_col].replace('\\', '/').replace('M:/bird identification/Quarantine recordings Tel Aviv (Yoel)/', '')
            # Remove 3rd sub-directory if appears
            directory = '/'.join(directory.split('/')[:2])
            if as_path:
                file_path = Path(base_path, directory, row[filename_col] + '.WAV')
            else:
                file_path = os.path.join(base_path, directory, row[filename_col]+'.WAV')
            file_list.append(file_path)
        # remove duplicates
        file_list = list(set(file_list))
        return file_list

    def confirm_files(self, base_path):
        """
        confirm that every file on list generated by get_file_list is indeed on records_list
        :param base_path: path where records are saved
        :return:
        """
        records_list = get_records_list(base_path)
        marked_list = self.get_file_list(base_path)
        confirmed_list = []
        for record_file in marked_list:
            confirmed_list.append(record_file in records_list)
        self.confirmed_list = confirmed_list
        # cl = np.array(confirmed_list)
        # cl.astype(int).sum()
        # self.marked_files[confirmed_col] = confirmed_list

    def gen_unify_species_column(self):
        """
        adds new column with all marked species of a single row
        """
        column_indexes = []
        # Species column indexes
        for column_str in species_columns:
            column_indexes.append(int(np.where(self.marked_files.columns == column_str)[0]))
        self.marked_files[unified_species_col] = self.marked_files.apply(gen_unify_label_str(column_indexes), axis=1)

    def generate_label_information(self, csv_dst=None):
        """
        generate SSAST type labels based on unify species column
        saves info to self params and optional to file
        :param csv_dst: (optional) path to save csv labels file
        """
        if unified_species_col not in self.marked_files.columns:
            self.gen_unify_species_column()
        species_list = self.marked_files[unified_species_col].to_list()
        flat_list = list()
        for sub_list in species_list:
            flat_list += sub_list
        species = set(flat_list)
        # Find all specie types by unified species column
        self.species = set(species)
        # Generate labels by speechcommands_class_labels_indices,csv example
        # Add "NoCall" labels
        zfill_n = math.ceil(math.log10(len(self.species) + 3))
        labels = [label_prefix + str(k).zfill(zfill_n) for k in range(len(self.species) + 1)]
        self.species_str = [no_call_label] + list(self.species)
        self.labels_dict = dict(zip(self.species_str, labels))
        self.labels_dict_inv = dict(zip(labels, self.species_str))
        self.labels_df = pd.DataFrame(list(self.labels_dict_inv.items()), columns=labels_csv_columns[1:])
        # Save to CSV
        if csv_dst is not None:
            self.labels_df.to_csv(csv_dst, index_label='index')

    def generate_json(self, json_dst_dir=None, part_list_path=None, wav_record_path=None,
                  partial=None, low_count_label_filter=None, string_filter_list=None):
        """
        generate train/validate/test jsons with labels by part_list json file and marked_files label data
        :param json_dst_dir:  target directory for the generated json files
        :param part_list_path: source path for the part_list json file (records divided to train/valid/test)
        :param wav_record_path: path to the records location
        :param partial: if not None should be dict with partial value per train/valid/test
        :param low_count_label_filter: if not None should be integer for filter threshold
        :param string_filter_list: list of strings, if label has one of the strings it should be filtered out
        :return: None
        """
        if self.labels_dict is None:
            self.generate_label_information()
        if self.labels_count is None:
            self.count_labels()
        if part_list_path is not None:
            with open(part_list_path, 'r') as f:
                part_list_dict = json.load(f)
        else:
            print('part_list path is needed')
            return
        if wav_record_path is None:
            print('wav file records path path is needed')
            return
        # Generate low count label filter list
        black_list = []
        if low_count_label_filter is not None:
            # for every label name
            for label in self.labels_count.keys():
                if self.labels_count[label] <= low_count_label_filter:
                    black_list.append(label)
        # add string filter labels from string list
        if string_filter_list is not None:
            for label in self.labels_count.keys():
                for filter_string in string_filter_list:
                    if filter_string in label:
                        black_list.append(label)
        self.json_black_list = black_list
        print('Generating with following blacklist:')
        print(black_list)
        # Generate file labels
        # zfill_n = math.ceil(math.log10(file_length_seconds + 3))
        for split_str in data_split_str_list:
            print(f'Processing {split_str} files')
            file_data_list = []
            for file_path in part_list_dict[split_str]:
                file_path_p = Path(file_path)
                file_stem = file_path_p.stem
                file_df = self.marked_files.loc[self.marked_files[filename_col] == file_stem]
                file_labels_list = ['' for _ in range(file_length_seconds)]
                # Fill labels for every second in the file on file_labels_list
                # Add 1sec file to the file_data_list
                for t_sec in range(file_length_seconds):
                    # Adding device number to file name
                    device = recorder_device_from_original_path(file_path)
                    one_sec_file_name = device + '_' + file_path_p.stem + '_' + str(t_sec) + file_path_p.suffix
                    one_sec_file_path = Path(wav_record_path, split_str, one_sec_file_name)
                    if t_sec in file_df[time_in_file_col].values:
                        label_list = file_df.loc[file_df[time_in_file_col] == t_sec][unified_species_col].values[0]
                        # Append only labels that are not on the low count filter list
                        file_labels_list[t_sec] = ','.join([self.labels_dict[label] for label in label_list if
                                                            label not in black_list])
                    else:
                        file_labels_list[t_sec] = self.labels_dict[no_call_label]
                    file_data = {
                        'wav': str(one_sec_file_path),
                        'labels': file_labels_list[t_sec]
                    }
                    # Append only records with non empty label lists (after low count filter)
                    if len(file_labels_list[t_sec]) > 0:
                        file_data_list.append(file_data)
            partial_str = ''
            if partial is not None:
                num_to_save = int(len(file_data_list) * partial[split_str])
                file_data_list = file_data_list[:num_to_save]
                partial_str = f'_partial_{num_to_save}'
            low_count_str = ''
            if low_count_label_filter is not None:
                low_count_str = f'_low_count_filter_{low_count_label_filter}'
            dst_file_name = f'birdcalls_{split_str}_data{partial_str}{low_count_str}.json'
            dst_file = Path(json_dst_dir, dst_file_name)
            with open(dst_file, 'w') as f:
                json.dump({'data': file_data_list}, f, indent=1)
        # Save label mask tensor
        if low_count_label_filter is not None:
            self.label_mask = [int(label not in black_list) for label in self.labels_df['display_name'].values]
            # self.label_mask = torch.tensor(self.label_mask)
            dst_mask_file = Path(json_dst_dir, f'label_mask{low_count_str}.json')
            with open(dst_mask_file, 'w') as f:
                json.dump({'label_mask': self.label_mask}, f, indent=1)

    def count_labels(self):
        if unified_species_col not in self.marked_files.columns:
            self.gen_unify_species_column()
        unified_species = self.marked_files[unified_species_col].values
        all_labels = []
        for specie_list in unified_species:
            all_labels += specie_list
        labels_count = {}
        for label in all_labels:
            if label in labels_count.keys():
                labels_count[label] += 1
            else:
                labels_count[label] = 1
        self.labels_count = labels_count

    def label_filter_statistics(self, count_th=100):
        if self.labels_count is None:
            self.count_labels()
        labels_in = 0
        labels_out = 0

        self.count_labels()
        for label in self.labels_count.keys():
            if self.labels_count[label] > count_th:
                print(f'{label}={self.labels_count[label]}')
                labels_in += 1
            else:
                labels_out += 1

        print()
        print(f'labels_in={labels_in}')
        print(f'labels_out={labels_out}')

   
    # -------------------------------------------------------------
    # dataset preparation for Bird - no Bird 2 label binary classification task
    # -------------------------------------------------------------
    def generate_binary_label_information(self, b_csv_dst=None, csv_dst=None):
        """
        generate SSAST type labels based on unify species column
        saves info to self params and optional to file
        :param csv_dst: (optional) path to save csv labels file
        """
        if unified_species_col not in self.marked_files.columns:
            self.gen_unify_species_column()
        
        # not used this block
        species_list = self.marked_files[unified_species_col].to_list()
        flat_list = list()
        for sub_list in species_list:
            flat_list += sub_list
        species = set(flat_list)
        # Find all specie types by unified species column
        self.species = set(species)
        # Generate labels by speechcommands_class_labels_indices,csv example
        # Add "NoCall" labels

        # original
        zfill_n = math.ceil(math.log10(len(self.species) + 3))
        labels = [label_prefix + str(k).zfill(zfill_n) for k in range(len(self.species) + 1)]
        self.species_str = [no_call_label] + list(self.species)
        self.labels_dict = dict(zip(self.species_str, labels))
        self.labels_dict_inv = dict(zip(labels, self.species_str))
        self.labels_df = pd.DataFrame(list(self.labels_dict_inv.items()), columns=labels_csv_columns[1:])
        # Save to CSV
        if csv_dst is not None:
            self.labels_df.to_csv(csv_dst, index_label='index')        

        b_zfill_n = math.ceil(math.log10(1 + 3))
        b_labels = [label_prefix + str(k).zfill(b_zfill_n) for k in range(1 + 1)]
        self.b_species_str = [no_call_label] + [bird_call_label]
        self.b_labels_dict = dict(zip(self.b_species_str, b_labels))
        self.b_labels_dict_inv = dict(zip(b_labels, self.b_species_str)) # zip( [/m/bird0, /m/bird1], [NoCall, BirdCall] ) 
        self.b_labels_df = pd.DataFrame(list(self.b_labels_dict_inv.items()), columns=labels_csv_columns[1:])
        # Save to CSV
        if b_csv_dst is not None:
            self.b_labels_df.to_csv(b_csv_dst, index_label='index')


    def generate_binary_label_json(self, json_dst_dir=None, part_list_path=None, wav_record_path=None,
                      partial=None, low_count_label_filter=None, binary_label=False):
        """
        generate train/validate/test jsons with labels by part_list json file and marked_files label data
        :param json_dst_dir:  target directory for the generated json files
        :param part_list_path: source path for the part_list json file (records divided to train/valid/test)
        :param wav_record_path: path to the records location
        :param partial: if not None should be dict with partial amount per train/valid/test
        :param low_count_label_filter: if not None should be integer for filter threshold
        :return: None
        """
        if self.labels_dict is None:
            self.generate_label_information() # generate_binary_label_information
        if self.labels_count is None:
            self.count_labels()
        if part_list_path is not None:
            with open(part_list_path, 'r') as f:
                part_list_dict = json.load(f)
        else:
            print('part_list path is needed')
            return
        if wav_record_path is None:
            print('wav file records path path is needed')
            return
        # Generate low count label filter list
        low_count_list = []
        if low_count_label_filter is not None:
            for label in self.labels_count.keys():
                if self.labels_count[label] <= low_count_label_filter:
                    low_count_list.append(label)
        # Generate file labels
        # zfill_n = math.ceil(math.log10(file_length_seconds + 3))
        for split_str in data_split_str_list:
            print(f'Processing {split_str} files')
            file_data_list = []
            for file_path in part_list_dict[split_str]:
                file_path_p = Path(file_path)
                file_stem = file_path_p.stem
                file_df = self.marked_files.loc[self.marked_files[filename_col] == file_stem]
                file_labels_list = ['' for _ in range(file_length_seconds)]
                # Fill labels for every second in the file on file_labels_list
                # Add 1sec file to the file_data_list
                for t_sec in range(file_length_seconds):
                    # Adding device number to file name
                    device = recorder_device_from_original_path(file_path)
                    one_sec_file_name = device + '_' + file_path_p.stem + '_' + str(t_sec) + file_path_p.suffix
                    one_sec_file_path = Path(wav_record_path, split_str, one_sec_file_name)
                    if t_sec in file_df[time_in_file_col].values:
                        label_list = file_df.loc[file_df[time_in_file_col] == t_sec][unified_species_col].values[0]
                        # Append only labels that are not on the low count filter list
                        file_labels_list[t_sec] = ','.join([self.labels_dict[label] for label in label_list if
                                                            label not in low_count_list])
                        if (len(file_labels_list[t_sec]) > 0) and (binary_label == True):
                             file_labels_list[t_sec] = self.b_labels_dict[bird_call_label]
                    else:
                        if (binary_label == True):
                            file_labels_list[t_sec] = self.b_labels_dict[no_call_label]
                        else:
                            file_labels_list[t_sec] = self.labels_dict[no_call_label]
                    file_data = {
                        'wav': str(one_sec_file_path),
                        'labels': file_labels_list[t_sec]
                    }
                    # Append only records with non empty label lists (after low count filter)
                    if len(file_labels_list[t_sec]) > 0:
                        file_data_list.append(file_data)
            partial_str = ''
            if partial is not None:
                num_to_save = int(len(file_data_list) * partial[split_str])
                file_data_list = file_data_list[:num_to_save]
                partial_str = f'_partial_{num_to_save}'
            low_count_str = ''
            if low_count_label_filter is not None:
                low_count_str = f'_low_count_filter_{low_count_label_filter}'
            dst_file_name = f'birdcalls_{split_str}_data{partial_str}{low_count_str}.json'
            dst_file = Path(json_dst_dir, dst_file_name)
            with open(dst_file, 'w') as f:
                json.dump({'data': file_data_list}, f, indent=1)
        # Save label mask tensor
        if low_count_label_filter is not None:
            self.label_mask = [int(label not in low_count_list) for label in self.labels_df['display_name'].values]
            # self.label_mask = torch.tensor(self.label_mask)
            dst_mask_file = Path(json_dst_dir, f'label_mask{low_count_str}.json')
            with open(dst_mask_file, 'w') as f:
                json.dump({'label_mask': self.label_mask}, f, indent=1)



def gen_unify_label_str(column_indexes):
    """
    for use with apply function on marked files DataFram
    :param column_indexes: species column indexes
    :return:
    """
    def unify_label_str(row):
        labels = []
        for index in column_indexes:
            if type(row[index]) == str:
                labels.append(standardize_label(row[index]))
        return labels
    return unify_label_str


def recorder_device_from_original_path(path):
    path = Path(path)
    device_dir = str(path.parent).split('/')[-1]
    device_num_str = device_dir.split(' ')[-1]
    return device_num_str


def standardize_label(label_str):
    """
    standardize specie string
    """
    std_label = label_str[1:]+'?' if label_str.startswith('?') else label_str
    std_label = std_label.title()
    std_label = std_label.replace(' ', '')
    return std_label


records_path = '/content/gdrive/My Drive/DeepLearning/bird_records/'
def get_records_list(records_path):
    """
    get list of saved wav records on a path
    """
    result = list(Path(records_path).rglob("*.[wW][aA][vV]"))
    return result


record_list_path = '/content/gdrive/My Drive/DeepLearning/project/records_list.pickle'
def save_data(data_dict, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)


def load_data(load_path):
    with open(record_list_path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def split_wav_file(file_path, dst_dir_path, add_device=True, use_file_duration=False):
    """
    Split file_length_seconds wave file to 1sec files
    Using pydub module
    """
    file_path = Path(file_path)
    # print(f'@split_wav_file file_path={file_path}')
    # print(f'@split_wav_file dst_dir_path={dst_dir_path}')
    wav_file = AudioSegment.from_file(file=file_path, format="wav")
    # Add device number if add_device selected
    device_str = ''
    if add_device:
        file_device_dir = str(Path(file_path).parent).split('/')[-1]
        if 'Device' in file_device_dir:
            device_str = file_device_dir.split(' ')[-1] + '_'
        else:
            device_str = '0_'
    duration = int(wav_file.duration_seconds)
    if not use_file_duration:
        split_length = file_length_seconds
        if duration < file_length_seconds:
            print(f'File {file_path} is less than {file_length_seconds} seconds. Not processed')
            return
    else:
        split_length = duration
        print(f'Splitting {file_path}, duration {duration}')
    t_slice = 1000
    for t_start in range(0, split_length * 1000, t_slice):
        slice_audio = wav_file[t_start:t_start + t_slice]
        slice_filename = device_str + file_path.stem + f'_{int(t_start / 1000)}' + file_path.suffix
        slice_path = Path(dst_dir_path, slice_filename)
        slice_audio.export(slice_path, format="wav")


class DatasetGenerator:
    """
    Generate SSAST records dataset from bird calls records
    receive list of records and a destination dir
    randomly split file list to train/validate/test
    split each file from to 1 second single files
    save 1 second files to directories of train/validate/test
    save file name with recorder number and original file name
    """
    def __init__(self, file_list, dst_dir_path):
        self.file_list = file_list
        self.dst_dir_path = dst_dir_path
        self.train_list = None
        self.valid_list = None
        self.test_list = None
        self.split_dict = None

    def create_dataset_from_file_list(self, test_part=0.2, validate_part=0.1):  # , json_dst_path=None):
        # split files list to train/validate/test lists
        train_valid_list, test_list = train_test_split(self.file_list, test_size=test_part, shuffle=True, random_state=42)
        train_list, valid_list = train_test_split(train_valid_list, test_size=validate_part, shuffle=True, random_state=42)
        # save lists
        parts_path_list = data_split_str_list
        split_lists = [train_list, valid_list, test_list]
        self.train_list = train_list
        self.valid_list = valid_list
        self.test_list = test_list
        # lists_save_path = Path(self.dst_dir_path, 'part_lists.pickle')
        parts_path_list_str = []
        for posix_list in split_lists:
            parts_path_list_str.append([str(record_path) for record_path in posix_list])
        parts_dict = dict(zip(parts_path_list, parts_path_list_str))
        with open(Path(self.dst_dir_path, 'part_lists.json'), 'w') as f:
            json.dump(parts_dict, f)
        with open(Path(self.dst_dir_path, 'part_lists.pkl'), 'wb') as f:
            pickle.dump(parts_dict, f, pickle.HIGHEST_PROTOCOL)
        # Gnerate time split files by train/validate/test lists
        self.split_dict = {}
        for part_list, part_name in zip(split_lists, data_split_str_list):
            self.split_dict[part_name] = part_list
            print(f'Now processing {part_name} part')
            print(f'Length {len(part_list)}')
            part_dst_path = Path(self.dst_dir_path, part_name)
            # Create directory if doesn't exist
            if not os.path.exists(part_dst_path):
                os.mkdir(part_dst_path)
            # Split all record files on the part list and save them on the part (train/validate/test) directory
            for record_file in tqdm(part_list):
                split_wav_file(record_file, part_dst_path)
        # if json_dst_path is not None:
        #     with open(json_dst_path, 'w') as f:
        #         json.dump(self.split_dict, f)

    def create_json_file(self, dst_path):
        with open(dst_path, 'w') as f:
            json.dump(self.split_dict, f)


def update_dataset_jsons(json_file_list, path_replace):
    path_from = path_replace['from']
    path_to = path_replace['to']
    for json_file in json_file_list:
        # data_list = []
        with open(json_file, 'r') as f:
            dataset_data = json.load(f)['data']
        for sample_data in dataset_data:
            sample_data["wav"] = sample_data["wav"].replace(path_from, path_to)
        new_json_file = Path(json_file)
        new_json_file = Path(path_to, new_json_file.stem + '_u' + new_json_file.suffix)
        with open(new_json_file, 'w') as f:
            json.dump({'data': dataset_data}, f, indent=1)


def gen_label_samples_dict(json_src):
    # Read source json
    with open(json_src, 'r') as f:
        file_data_list = json.load(f)['data']
    # List samples by labels
    label_samples_dict = {}
    for k, sample in enumerate(file_data_list):
        record_labels = label_str_tolist(sample['labels'])
        for label in record_labels:
            if label in label_samples_dict.keys():
                label_samples_dict[label].append(k)
            else:
                label_samples_dict[label] = [k]
    return label_samples_dict

def resampling(json_src, json_dst, replicate_count_threshold, minimum_th):
    """
        Replicate files entries with labels that has low count
        Replication ration by inverse ratio to label count
        :param replicate_count_threshold: Replicate only for labels with count lower than the threshold. replicate to threshold.
        :param minimum_th: Replicate only samples that are above the minimum filter count
    """
    # Read source json
    with open(json_src, 'r') as f:
        file_data_list = json.load(f)['data']
    label_samples_dict = gen_label_samples_dict(json_src)
    # Print initial label count and threshold
    print('Initial label count')
    for label in label_samples_dict.keys():
        print(f'{label}={len(label_samples_dict[label])}')
    print('')
    # generate expand lists
    expand_dict = {}
    for label in label_samples_dict.keys():
        samples_list = label_samples_dict[label]
        list_len = len(samples_list)
        # expand only when count is above the minimum filter minimum threshold
        if list_len > minimum_th:
            # expand only when count is below replicate count threshold
            if list_len < replicate_count_threshold:
                # Add (replicate_count_threshold - list_len sample) indexes to sample list
                full_replicates = int((replicate_count_threshold-list_len) / list_len)
                partial_replicate = replicate_count_threshold % list_len
                samples_expand_list = samples_list * full_replicates + samples_list[:partial_replicate]
                expand_dict[label] = samples_expand_list
    # to jason file data list
    new_file_data_list = file_data_list
    for label in expand_dict.keys():
        new_file_data_list += [file_data_list[idx] for idx in expand_dict[label]]
    # generate destenation json
    with open(json_dst, 'w') as f:
        json.dump({'data': new_file_data_list}, f, indent=1)
    # Read results json and print new label count
    label_samples_dict = gen_label_samples_dict(json_dst)
    # Print initial label count and threshold
    print('Resampled label count')
    for label in label_samples_dict.keys():
        print(f'{label}={len(label_samples_dict[label])}')

def gen_pos_weight(json_src, labels_csv_path):
    # Read source json
    label_samples_dict = gen_label_samples_dict(json_src)
    labels = pd.read_csv(labels_csv_path)
    labels_count = torch.zeros([len(labels)])
    for k, label in enumerate(labels['mid']):
        if label in label_samples_dict.keys():
            labels_count[k] = len(label_samples_dict[label])
    pos_weight = labels_count.max() / labels_count
    pos_weight[torch.where(labels_count == 0)] = 0
    return pos_weight



# records_list = get_records_list(records_path)
# records_list_d = {'records_list': records_list}
# save_data(records_list, record_list_path)
# rl = load_data(record_list_path)['records_list']

# mf.confirm_files(records_path, records_list)
# wp.split_wav_file(marked_list[10], labeld_data_path)

# data_generator = wp.DatasetGenerator(marked_list, labeld_data_path)
# data_generator.create_dataset_from_file_list(test_part=0.2, validate_part=0.1)

# mf.generate_label_information('birdcalls_class_labels_indices.csv')






