
from pathlib import Path

from wav_processing import MarkedFiles, DatasetGenerator, data_split_str_list
from data_validator import Validator


def prepare_dataset(excel_path, orig_records_path, dst_dir_path,
                    partial=None, low_count_label_filter=None, string_filter_list=None):
    # Load data from marking excel
    print(f'Loading excel data from {excel_path}')
    mf = MarkedFiles(excel_path)
    # Generate labels csv
    csv_dst = Path(dst_dir_path, 'birdcalls_class_labels_indices.csv')
    print(f'Generate labels csv {csv_dst}')
    mf.generate_label_information(csv_dst=csv_dst)
    # Generate dataset 1sec records files splitted to train/test/validate directories
    print()
    file_list = mf.get_file_list(orig_records_path)
    print(f'Found {len(file_list)} different files on marking excel')
    print(f'Splitting original records from {orig_records_path} to 1sec slices ')
    print(f'Arrange files in train/test/validate directories on {dst_dir_path}')
    data_gen = DatasetGenerator(file_list, dst_dir_path)
    second_json_dst_path = Path(dst_dir_path, 'part_lists_B.json')
    data_gen.create_dataset_from_file_list()  #  (json_dst_path=second_json_dst_path)
    # Generate label jsons
    print()
    print('Generate json files for SSAST in {dst_dir_path}')
    if partial is not None:
        print('json files will show only {partial} of the data')
    mf.generate_json(json_dst_dir=dst_dir_path,
                     part_list_path=Path(dst_dir_path, 'part_lists.json'),
                     wav_record_path=dst_dir_path,
                     partial=partial,
                     low_count_label_filter=low_count_label_filter,
                     string_filter_list= string_filter_list)


def validate_data(excel_path, labels_csv_path, part_list_path, labeled_path_dict):
    v = Validator(excel_path, labels_csv_path, part_list_path, labeled_path_dict)
    print('Check occurrences on part list json')
    v.check_part_list_occurrences()
    print('Check occurrences on train/validate/test jsons')
    v.check_labeled_jsons_occurrences()
    print('Validate labels on train/validate/test jsons')
    v.validate_labels_on_jsons()



# -------------------------------------------------------------
# dataset preparation for Bird - no Bird 2 label binary classification task
# -------------------------------------------------------------

def prepare_binary_dataset(excel_path, orig_records_path, dst_dir_path,
                    partial=None, low_count_label_filter=None):
    # Load data from marking excel
    print(f'Loading excel data from {excel_path}')
    mf = MarkedFiles(excel_path)
    # Generate labels csv
    csv_dst = Path(dst_dir_path, 'birdcalls_class_labels_indices.csv')    
    binary_csv_dst = Path(dst_dir_path, 'birdcalls_2_class_labels_indices.csv')
    print(f'Generate labels csv {csv_dst}')
    mf.generate_binary_label_information(b_csv_dst=binary_csv_dst, csv_dst=csv_dst)
    # Generate dataset 1sec records files splitted to train/test/validate directories
    print()
    file_list = mf.get_file_list(orig_records_path)
    print(f'Found {len(file_list)} different files on marking excel')
    print(f'Splitting original records from {orig_records_path} to 1sec slices ')
    print(f'Arrange files in train/test/validate directories on {dst_dir_path}')
    data_gen = DatasetGenerator(file_list, dst_dir_path)
    second_json_dst_path = Path(dst_dir_path, 'part_lists_B.json')
    data_gen.create_dataset_from_file_list()  #  (json_dst_path=second_json_dst_path)
    # Generate label jsons
    print()
    print('Generate json files with binary labels for SSAST in {dst_dir_path}')
    if partial is not None:
        print('json files will show only {partial} of the data')
    mf.generate_binary_label_json(json_dst_dir=dst_dir_path,
                     part_list_path=Path(dst_dir_path, 'part_lists.json'),
                     wav_record_path=dst_dir_path,
                     partial=partial,
                     low_count_label_filter=low_count_label_filter,
                     binary_label=True)
