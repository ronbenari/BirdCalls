# exec(open("./analyze_ssast_results.py").read())

import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
import csv

# run_name = '2022_09_30_filter_10_Mixing 0_6'
# base_path = Path(results_dir, run_name)
# arg_filename = 'args.pkl'
# pred_filename = 'predictions_valid_set.csv'
# target_filename = 'target_test.csv'
# labels_filename = 'birdcalls_class_labels_indices.csv'
# csv_dst_filename = 'results_anlz_stats.csv'
# csv_dst_summary = Path(results_dir, 'summary.csv')

# pred_path = Path(base_path, 'predictions', pred_filename)
# target_path = Path(base_path, 'predictions', target_filename)
# args_path = Path(base_path, arg_filename)
# labels_path = Path(base_path, labels_filename)
# csv_dst_path = Path(base_path, csv_dst_filename)


# Jasons Path
# json_dir = Path(results_dir, 'labeled data 2022 09 14')
# train_json = Path(json_dir, 'birdcalls_train_data_low_count_filter_100.json')
# validate_json = Path(json_dir, 'birdcalls_validate_data_low_count_filter_100.json')
# test_json = Path(json_dir, 'birdcalls_test_data_low_count_filter_100.json')

# Non-active labels filter values
std_th = 1e-2
mean_dist_th = 0.05


class SSASTResults():
    def __init__(self, pred_path, target_path, args_path=None, labels_path=None):
        self.pred_path = pred_path
        self.target_path = target_path
        self.args_path = args_path
        self.labels_path = labels_path

        self.pred = pd.read_csv(pred_path)
        self.target = pd.read_csv(target_path)
        self.labels = None
        if labels_path is not None:
            self.labels = pd.read_csv(labels_path)
        self.args = None
        if args_path is not None:
            with open(args_path, 'rb') as file:
                # Call load method to deserialze
                try:
                    self.args = pickle.load(file)
                except:
                    print(f'Error loading args f{file}')
        self.labels_list = None
        self.label_name_to_index = None
        self.label_code_to_index = None
        self.stats_dict = None
        self.label_names = None

    def make_labels_list(self):
        if self.labels is None:
            print('No Labels found')
            return
        labels_list = []
        label_names = []
        label_name_to_index = {}
        label_code_to_index = {}
        for k in range(len(self.labels)):
            [label, label_name] = self.labels.iloc[k].values[1:3].tolist()
            labels_list.append({'label': label,
                                'label_name': label_name})
            label_name_to_index[label_name] = k
            label_code_to_index[label] = k
            label_names.append(label_name)
        self.labels_list = labels_list
        self.label_name_to_index = label_name_to_index
        self.label_code_to_index = label_code_to_index
        self.label_names = label_names

    def label_stats(self, label_name, th=0.5):
        if self.label_name_to_index is None:
            self.make_labels_list()
        label_index = self.label_name_to_index[label_name]
        label_target = self.target.values[:, label_index]
        label_pred_raw = self.pred.values[:, label_index]
        label_pred = (self.pred.values[:, label_index] > th).astype(int)
        total_calls = label_target.sum()
        acc = (label_target == label_pred).mean()
        precision = precision_score(label_target, label_pred, zero_division=0)
        recall = recall_score(label_target, label_pred, zero_division=0)
        record_pos_count_target = [label_target[k*30:k*30+30].sum() for k in range(int(len(label_target)/30))]
        record_pos_count_target = np.array(record_pos_count_target)
        record_pos_count_pred = [label_pred[k * 30:k * 30 + 30].sum() for k in range(int(len(label_pred) / 30))]
        record_pos_count_pred = np.array(record_pos_count_pred)
        try:
            auc_score = roc_auc_score(label_target, label_pred_raw)
        except:
            auc_score = 0
        # best th for maximizing f1 score
        f1_score_array = []
        epsilon = 1e-8
        th_range = np.arange(0.01, 1.0, 0.01)
        for f1_th in th_range:
            f1_label_pred = (self.pred.values[:, label_index] > f1_th).astype(int)
            f1_precision = precision_score(label_target, f1_label_pred, zero_division=0)
            f1_recall = recall_score(label_target, f1_label_pred, zero_division=0)
            f1_score = 2 * f1_precision * f1_recall / (f1_precision + f1_recall + epsilon)
            f1_score_array.append(f1_score)
        f1_score_array = np.asarray(f1_score_array)
        f1_max_index = f1_score_array.argmax()
        res = {'acc': acc,
               'precision': precision,
               'recall': recall,
               'total_calls': total_calls,
               'auc_score': auc_score,
              'record_pos_count_target': record_pos_count_target,
              'record_pos_count_pred': record_pos_count_pred,
              'f1_score_array':f1_score_array,
               'f1_score_max_th': th_range[f1_max_index],
               'f1_score_max_value': f1_score_array[f1_max_index]
        }
        return res

    def calc_stats(self, th=0.5):
        if self.label_name_to_index is None:
            self.make_labels_list()
        stats_dict = {}
        for label in self.labels_list:
            label_name = label['label_name']
            label_index = self.label_name_to_index[label_name]
            # If pred variance is about zero and it is around 0.5 it was not an active label - filterring it out
            label_pred = self.pred.values[:, label_index]
            pred_std = label_pred.std()
            pred_mean = label_pred.mean()
            mean_0_50_dist = np.abs(pred_mean-0.5)
            if (pred_std < std_th) and (mean_0_50_dist < mean_dist_th):
                continue
            stats_dict[label_name] = self.label_stats(label_name, th)
        self.stats_dict = stats_dict
        # calc average precision


    def stats_to_csv(self, dst):
        if self.stats_dict is None:
            self.calc_stats()
        df = pd.DataFrame.from_dict(self.stats_dict)
        df.to_csv(dst)

    def get_target_pred_by_label_name(self, label_name):
        if self.label_name_to_index is None:
            self.make_labels_list()
        label_index = self.label_name_to_index[label_name]
        label_target = self.target.values[:, label_index]
        label_pred = self.pred.values[:, label_index]
        return label_target, label_pred

    def json_to_csv(self, src_json, dst_csv=None):
        if self.label_name_to_index is None:
            self.make_labels_list()
        with open(src_json, 'r') as f:
            records_data = json.load(f)['data']
        # convert json to 0-1 csv with bird names title, seconds column
        active_calls = np.zeros([len(records_data), len(self.labels)])
        index_values = []
        for row, record_item in enumerate(records_data):
            file_name = record_item["wav"]
            labels = record_item["labels"]
            labels = labels.split(',')
            try:
                active_labels_indexes = [self.label_code_to_index[lbl] for lbl in labels]
            except:
                active_labels_indexes = []
                print(f'Error reading labels "{labels}" on row {row}')
            row_indexes = [row] * len(active_labels_indexes)
            active_calls[row_indexes, active_labels_indexes] = 1
            index_values.append(file_name + f'_{row % 30}')
        # To data frame and to csv
        df = pd.DataFrame(data=active_calls,
                          index=index_values,
                          columns=self.label_names)
        if dst_csv is None:
            dst_csv = Path(Path(src_json).parent, Path(src_json).stem + '.csv')
        print(f'Saving to {dst_csv}')
        self.df_json = df
        df.to_csv(dst_csv)



class StatsCsv():
    stat_row = {'acc': 0, 'precision': 1, 'recall': 2, 'total_calls': 3, 'auc_score': 4,
                'record_pos_count_target': 5, 'record_pos_count_pred': 6,
                'f1_score_array' :7, 'f1_score_max_th': 8, 'f1_score_max_value': 9}
    def __init__(self, csv_src):
        self.csv_src = csv_src
        self.name = str(Path(csv_src).parent).split('/')[-1]
        self.df = pd.read_csv(csv_src)
        # Save row/column data without title cell
        self.labels = self.df.columns[1:]
        self.acc = self.df.iloc[self.stat_row['acc']].values[1:].astype(float)
        self.precision  = self.df.iloc[self.stat_row['precision']].values[1:].astype(float)
        self.recall = self.df.iloc[self.stat_row['recall']].values[1:].astype(float)
        self.f1_score_max_th = self.df.iloc[self.stat_row['f1_score_max_th']].values[1:].astype(float)
        self.f1_score_max_value = self.df.iloc[self.stat_row['f1_score_max_value']].values[1:].astype(float)
        self.total_calls  = self.df.iloc[self.stat_row['total_calls']].values[1:].astype(float)
        self.auc_score = self.df.iloc[self.stat_row['auc_score']].values[1:].astype(float)
        self.acc_by_calls = self.acc @ self.total_calls / self.total_calls.sum()
        self.precision_by_calls = self.precision @ self.total_calls / self.total_calls.sum()
        self.recall_by_calls = self.recall @ self.total_calls / self.total_calls.sum()
        self.auc_score_by_calls = self.auc_score @ self.total_calls / self.total_calls.sum()
        self.record_pos_count_target_all = self.df.iloc[self.stat_row['record_pos_count_target']].values[1:]
        self.record_pos_count_target_all = record_list_to_array(self.record_pos_count_target_all)
        self.record_pos_count_pred_all = self.df.iloc[self.stat_row['record_pos_count_pred']].values[1:]
        self.record_pos_count_pred_all = record_list_to_array(self.record_pos_count_pred_all)

    def find_worse_records(self, worse_count=3):
        # Score records by prediction error - higher error higher score
        score = rate_label_record_count(self.record_pos_count_pred_all, self.record_pos_count_target_all)
        # Find worse records per label
        idx = np.argsort(score, axis=1)[:, -worse_count:]
        unique, counts = np.unique(idx, return_counts=True)
        max_indexes_counts = np.where(counts == max(counts))[0]
        # worse records are found most times at the top error of the different labels
        worse_records = unique[max_indexes_counts]
        return worse_records


class DirectorySummary():
    def __init__(self, dir_path, filter=None, csv_dst_filename=csv_dst_filename):
        self.dir_path = dir_path
        self.filter = filter
        self.stat_csvs = []
        for res_dir_name in os.listdir(dir_path):
            if filter in res_dir_name:
                csv_path = Path(dir_path, res_dir_name, csv_dst_filename)
                print(f'Opening {csv_path}')
                try:
                    sc = StatsCsv(csv_path)
                    self.stat_csvs.append(sc)
                except:
                    print(f'Error opening {csv_path} !')

    def to_csv(self, dst_csv):
        summary_rows = []
        # Common data
        # Assuming all have same labels
        labels = self.stat_csvs[0].labels
        # Assuming total calls are same for all
        # Total Calls
        total_calls = self.stat_csvs[0].total_calls
        summary_rows.append(labels)
        summary_rows.append(total_calls)
        summary_rows.append([])
        # Common acc
        common_dict = {'Common Accuracy': 'acc_by_calls',
                       'Common Precision': 'precision_by_calls',
                       'Common Recall': 'recall_by_calls',
                       'Common AUC Score': 'auc_score_by_calls'}
        for common_metric in common_dict.keys():
            summary_rows.append(['Run Name', common_metric])
            for sc in self.stat_csvs:
                summary_rows.append([sc.name, getattr(sc, common_dict[common_metric])])  # getattr(sc, atrib_dict[attrib_name]).tolist()
            summary_rows.append([])
        # summary_rows.append(['Run Name', 'Common Accuracy'])
        # for sc in self.stat_csvs:
        #     summary_rows.append([sc.name, sc.acc_by_calls])
        # summary_rows.append([])
        # # Common Precision
        # summary_rows.append(['Run Name', 'Common Precision'])
        # for sc in self.stat_csvs:
        #     summary_rows.append([sc.name, sc.precision_by_calls])
        # summary_rows.append([])
        # # Common Recall
        # summary_rows.append(['Run Name', 'Common Recall'])
        # for sc in self.stat_csvs:
        #     summary_rows.append([sc.name, sc.recall_by_calls])
        # summary_rows.append([])

        # By labels
        atrib_dict={'labels': 'labels',
                    'accuracy': 'acc',
                    'precision': 'precision',
                    'recall': 'recall',
                    'auc_score': 'auc_score',
                    'f1_score_max_th': 'f1_score_max_th',
                    'f1_score_max_value': 'f1_score_max_value'}
        # Total calls
        summary_rows.append(['total_calls'])
        summary_rows.append([''] + self.stat_csvs[0].labels.to_list())
        summary_rows.append([''] + self.stat_csvs[0].total_calls.tolist())
        summary_rows.append([])
        # Attributes
        for attrib_name in atrib_dict.keys():
            summary_rows.append([attrib_name])
            summary_rows.append([''] + self.stat_csvs[0].labels.to_list())
            for sc in self.stat_csvs:
                summary_rows.append([sc.name] + getattr(sc, atrib_dict[attrib_name]).tolist())
            summary_rows.append([])
        self.summary_rows = summary_rows
        if dst_csv is not None:
            with open(dst_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerows(summary_rows)

# labels, recall, precision, acc, total_calls

def rate_label_record_count(record_pos_count_pred, record_pos_count_target):
    """ give a score to the accuracy of record label count, per record file """
    distance = record_pos_count_pred - record_pos_count_target
    score = np.abs(distance)
    return score

def plot_label_record_count(record_pos_count_pred, record_pos_count_target, title='', show=True, png_dst=None):
    fig, ax = plt.subplots()
    ax.plot(record_pos_count_pred, 'r', label='predictions')
    ax.plot(record_pos_count_target, 'b', label='target')
    ax.set_title(title)
    ax.set_xlabel('raw record files')
    ax.set_ylabel('calls per file')
    ax.legend()
    if show:
        plt.show()
    if png_dst is not None:
        fig.savefig(png_dst)

    # plt.figure()
    # plt.plot(record_pos_count_pred, 'r', label='predictions')
    # plt.plot(record_pos_count_target, 'b', label='target')
    # plt.title(title)
    # plt.xlabel('raw record files')
    # plt.ylabel('calls per file')
    # plt.legend()

def record_list_to_array(record_list_str):
    list_of_int_lists = []
    for list_str in record_list_str:
        list_str = list_str.replace('.', '')
        list_str = list_str.replace('\n', '')
        list_str = list_str.replace('[', '')
        list_str = list_str.replace(']', '')
        int_list = [int(element) for element in list_str.split(' ') if element != '']
        list_of_int_lists.append(int_list)
    return np.asarray(list_of_int_lists)

def get_record_name_from_json(json_file, record_idx):
    with open(json_file, 'r') as f:
        records_data = json.load(f)['data']
    return records_data[record_idx]["wav"]

def copy_csvs_to_directory(dir_path, target_path):
    if not os.path.isdir(target_path):
        print(f'{target_path} is not a directory!')
        return
    for dir_name in os.listdir(dir_path):
        src_path = Path(dir_path, dir_name)
        if os.path.isdir(src_path):
            csv_path = Path(src_path, csv_dst_filename)
            if os.path.isfile(csv_path):
                print(f'Copying {csv_path}')
                target_file = Path(target_path, csv_path.stem + f' - {dir_name}' + csv_path.suffix)
                shutil.copyfile(csv_path, target_file)

def make_csv_on_all_results(res_dir, m_labels_path, filter='', target_path=None,
                            m_pred_filename=pred_filename, m_csv_dst_filename=csv_dst_filename):
    for dir_name in os.listdir(res_dir):
        if filter in dir_name:
            src_path = Path(res_dir, dir_name)
            if os.path.isdir(src_path):
                pred_path = Path(src_path, 'predictions', m_pred_filename)
                if target_path is None:
                    target_path = Path(src_path, 'predictions', target_filename)
                args_path = Path(src_path, arg_filename)
                print(f'Opening {dir_name}')
                # print(f'pred_path: {pred_path}')
                # print(f'target_path: {target_path}')
                # print(f'args_path: {args_path}')
                # print(f'labels_path: {m_labels_path}')
                sr = SSASTResults(pred_path, target_path, args_path=args_path, labels_path=m_labels_path)
                csv_dst_path = Path(src_path, m_csv_dst_filename)
                print(f'Saving {csv_dst_path}')
                sr.stats_to_csv(csv_dst_path)


def analyze_dir(res_dir, split_name='test', filter=''):
    print(f'Analyzing directory {res_dir}')
    print(f'Split name "{split_name}"')
    # Generate csv summary of all directories
    res_labels_path = Path(res_dir, 'Labeled_Data', labels_filename)
    res_target_path = Path(results_dir, 'Labeled_Data', f'target_{split_name}_set.csv')
    res_pred_filename_dict = {'train': 'predictions_25.csv',
                              'valid': 'predictions_valid_set.csv',
                              'test': 'predictions_eval_set.csv'}
    res_pred_filename = res_pred_filename_dict[split_name]
    # res_target_name_dict = {'train': 'target_train_set.csv',
    #                         'valid': 'target_valid_set.csv',
    #                         'test': 'target_test_set.csv'}
    # res_target_path = res_target_path_dict[split_name]
    res_csv_dst_filename = f'results_anlz_stats_{split_name}.csv'
    print()
    print(f"Make csv's")
    make_csv_on_all_results(res_dir, m_labels_path=res_labels_path, filter=filter, target_path=res_target_path,
                            m_pred_filename=res_pred_filename, m_csv_dst_filename=res_csv_dst_filename)
    # Collect csvs data
    print()
    print(f"Collect data from all csv's")
    ds = DirectorySummary(res_dir, filter=filter, csv_dst_filename=res_csv_dst_filename)
    # save to summary csv
    summary_csv_file_name = f'summary_{split_name}_set.csv'
    save_path = Path(res_dir, summary_csv_file_name)
    print()
    print(f'Saving to csv file: {save_path}')
    ds.to_csv(save_path)


def compare_args(args_path1, args_path2):
    """ Comparing two args files, assuming same attributes on both args files"""
    with open(args_path1, 'rb') as file:
        args1 = pickle.load(file)
    with open(args_path2, 'rb') as file:
        args2 = pickle.load(file)
    for attrib_name in dir(args1):
        if not attrib_name.startswith('_'):
            attrib1 = getattr(args1, attrib_name)
            attrib2 = getattr(args2, attrib_name)
            if attrib1 != attrib2:
                print(f'{attrib_name}')
                print(f'args1 : {attrib1}')
                print(f'args2 : {attrib2}')
                print()

# sr = SSASTResults(pred_path, target_path, args_path=None, labels_path=labels_path)
# sr.calc_stats()
# sr.label_stats('Pycnonotus')
# label_target, label_pred = sr.get_target_pred_by_label_name('CommonMyna')

# for label in sr.stats_dict.keys():
#   anz.plot_label_record_count(sr.stats_dict[label]['record_pos_count_pred'], sr.stats_dict[label]['record_pos_count_target'], title=label)

# ['NoCall', 'Pycnonotus', 'HoodedCrow', 'LaughingDove', 'LesserWhitethroat', 'Rose-RingedParakeet', 'CommonBlackbird',
# 'MonkParakeet', 'PalestineSunbird', 'GracefulPrinia', 'GreatTit', 'EurasianCollaredDove', 'EuropeanRobin',
# 'EgyptianGoose', 'White-ThroatedKingfisher', 'CommonMyna'])

# src_json =  Path(base_path, 'jsons', 'birdcalls_test_data_low_count_filter_10.json')


# sc = StatsCsv(csv_dst_path)
# ds = DirectorySummary(results_dir, filter='2022_09_14')

# ds.to_csv('')
# ds.summary_rows

# make_csv_on_all_results(results_dir, labels_path, 'filter 100')

# Running on test set
# pred_filename = 'predictions_eval_set.csv'
# csv_dst_filename = 'results_anlz_stats_test_set.csv'
# make_csv_on_all_results(results_dir, labels_path, 'filter_10', target_path=Path(results_dir, 'Labeled_Data', 'target_test_set.csv'))
#
# ds.to_csv(Path(results_dir, 'summary_test_set.csv'))

# analyze_dir(res_dir, split_name='test')
# analyze_dir(res_dir, split_name='test', filter='model')
# analyze_dir(res_dir, split_name='valid', filter='model')
