import torch
from torch import nn

from wav_processing import split_wav_file
from pathlib import Path
from birds_run import eval_model, default_args
import pandas as pd
from ast_models import ASTModel
import os
import json
import pandas as pd


class ModelPred:
    """
    Predict bird call on wave file by stored model
    """
    # use_mask -> binary_prediction
    def __init__(self, model_size, model_files_path, output_path, binary_prediction=False):
        self.model_size = model_size
        self.model_files_path = model_files_path
        self.output_path = output_path
        print('Starting model predict')
        print(f'model files path: {self.model_files_path }')
        print(f'output files path: {self.output_path}\n')
        self.label_mask_path = Path(model_files_path, 'label_mask.json')
        if binary_prediction:
            self.score_thresholds_path = Path(model_files_path, 'score thresholds binary.csv')
            self.model_params_path = Path(model_files_path, 'audio_model_params_binary.pth')            
        else:
            self.score_thresholds_path = Path(model_files_path, 'score thresholds.csv')
            self.model_params_path = Path(model_files_path, 'audio_model_params.pth')
        # self.model_params_path = source_path
        # self.label_path = output_path

        self.use_mask = not binary_prediction
        args = set_args(model_size=self.model_size, model_files_path=model_files_path, output_path=output_path, binary_prediction=binary_prediction)
        self.args = args
        # Load SSL pretrained model
        print(f'Loading model {self.model_size} from {args.pretrained_mdl_path}')
        self.audio_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride,
                               tstride=args.tstride,
                               input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size,
                               pretrain_stage=False,
                               load_pretrained_mdl_path=args.pretrained_mdl_path)

        if not isinstance(self.audio_model, torch.nn.DataParallel):
            self.audio_model = torch.nn.DataParallel(self.audio_model)
        # Load fine tune model parameters
        print(f'Loading fine tune parameters from {self.args.fine_tuned_mdl_path}')
        self.audio_model.load_state_dict(torch.load(self.args.fine_tuned_mdl_path))
        self.audio_model.eval()

    def pred_file(self, records_path, split_dst_path=None):
        if split_dst_path is None:
            # split_dst_path = Path(file_path.parent, 'split')
            split_dst_path = Path(self.output_path, 'split')
        print(f'Saving split files on {split_dst_path}')
        # Generate directory if it doesn't exist
        os.makedirs(split_dst_path, exist_ok=True)
        # Split files to 1second files
        record_file_count = 0
        for filename in os.listdir(records_path):
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                wave_file_path = str(Path(records_path, filename))
                split_wav_file(file_path=wave_file_path, dst_dir_path=split_dst_path,
                               add_device=False, use_file_duration=True)
                record_file_count += 1
        print(f'Found total of {record_file_count} wave files on {records_path}')
        if (record_file_count == 0):
            print(f'WARNING ---------- !!! NO Found wave files in {records_path} ')
        # Create json for the 1sec files
        file_data_list = []
        one_sec_file_names = []
        if self.use_mask:
            dummy_label = '/m/birdc00' 
        else:
            dummy_label = '/m/birdc0'
        # sorted list of files  
        if len( os.listdir(split_dst_path) ) != 0:     
            sorted_listdir_split_dst_path = sort_list_of_wav_files( os.listdir(split_dst_path) )
        else: 
            sorted_listdir_split_dst_path = []    

        #for filename in os.listdir(split_dst_path):
        for filename in sorted_listdir_split_dst_path:
            # Setting dummy NoCall label (/m/birdc00)
            file_data = {
                'wav': str(Path(split_dst_path, filename)),
                'labels': dummy_label}   # pavel for binary trying instea of /m/birdc00
            file_data_list.append(file_data)
        print(f'Saved total of {len(file_data_list)} split wave files on {split_dst_path}')
        # save json
        json_path = self.args.data_eval
        print(f'Saving json file on {json_path}')
        with open(json_path, 'w') as f:
            json.dump({'data': file_data_list}, f, indent=1)
        # Run model prediction
        print('\nRunning model prediction')
        stats = eval_model(self.audio_model, self.model_params_path, self.args, no_target_stats=True)

    def process_model_pred(self):
        json_path = self.args.data_eval
        print(f'Read one second records list from {json_path}')
        with open(json_path, 'r') as f:
            file_data_list = json.load(f)['data']
        one_sec_file_list = [sample['wav'] for sample in file_data_list]
        orig_files_list, sec_in_files_list = split_file_list_to_orig_file_and_sec(one_sec_file_list)
        record_files_df = pd.DataFrame(list(zip(orig_files_list, sec_in_files_list)),
                                      columns=['Record Filename', 'Second in File'])
        # model_pred_scores_df = pd.read_csv(model_pred_file_path)
        label_df = pd.read_csv(self.args.label_csv)
        label_list = label_df['display_name'].values

        model_pred_file_path = Path(self.output_path, 'predictions', 'predictions_eval_set.csv')
        # if mask is used then only columns with mask == 1 are taken, in binary case not need mask 
        if self.use_mask: 
            with open(self.label_mask_path, 'r') as f:
                mask_list = json.load(f)['label_mask']
            valid_labels = pd.Series(label_list).loc[pd.Series(mask_list) == 1].values

            model_pred_scores_df = pd.read_csv(model_pred_file_path, index_col=False, names=label_list, usecols=valid_labels)
        else:
            valid_labels = pd.Series(label_list).values
            model_pred_scores_df = pd.read_csv(model_pred_file_path, index_col=False, names=label_list)

        # make Birds above th series
        score_thresholds_df = pd.read_csv(self.score_thresholds_path)
        score_thresholds_dict = score_thresholds_df[['display_name',
                                                     'Threshold']].set_index('display_name').to_dict(orient='index')
        above_th_df = model_pred_scores_df.copy()
        for col_name in above_th_df.columns:
            if col_name in score_thresholds_dict.keys():
                above_th_df[col_name] = (above_th_df[col_name] > score_thresholds_dict[col_name]['Threshold']).astype(
                    int)
        pred_labels_s = above_th_df.dot(above_th_df.columns + ',').str[:-1]
        # Add filenames and file time
        model_pred_scores_df = pd.concat([record_files_df, model_pred_scores_df], axis=1)
        # Save scores dataframe
        decorated_model_pred_path = Path(self.output_path, 'predictions', 'model scores.csv')
        print(f'Saving scores file to {decorated_model_pred_path}')
        model_pred_scores_df.to_csv(decorated_model_pred_path)
        # Savelabels per file dataframe
        pred_labels_df = pd.concat([record_files_df, pred_labels_s], axis=1)
        pred_labels_path = Path(self.output_path, 'predictions', 'model predicted labels.csv')
        print(f'Saving predicted labels file to {pred_labels_path}')
        pred_labels_df.to_csv(pred_labels_path)


def split_file_list_to_orig_file_and_sec(split_file_list):
    orig_files_list = []
    sec_in_files_list = []
    for filename in split_file_list:
        stem = Path(filename).stem
        sec_in_files_list.append(stem.split('_')[-1])
        orig_filename = '_'.join(stem.split('_')[:-1]) + Path(filename).suffix
        orig_files_list.append(orig_filename)
    return orig_files_list, sec_in_files_list

def sort_list_of_wav_files(split_file_list):
    split_file_list_without_wav = []
    split_file_list_sorted      = []
    for filename in split_file_list:
        if (filename.endswith('.WAV') or filename.endswith('.wav')):
            filename_no_suf = filename.split('.')[:1][0] # without [0] it's list
            split_file_list_without_wav.append(filename_no_suf)

    sec_in_file_list = []
    sorted_list = []
    file_name_prev = '_'.join(split_file_list_without_wav[0].split('_')[:-1])
    # split_file_list_without_wav.sort()  
    for file in split_file_list_without_wav:
        file_name = '_'.join( file.split('_')[:-1] )

        if (file_name != file_name_prev) or (file == split_file_list_without_wav[-1]):  # or file is the last element in a list  
            
            if (file == split_file_list_without_wav[-1]): # this is the end of the loop make append now
                sec_in_file_list.append( int( file.split('_')[-1]) )

            sec_in_file_list.sort()
            for sec in sec_in_file_list:
                sorted_list.append(file_name_prev + '_' + str(sec))     
            sec_in_file_list.clear()

        sec_in_file_list.append( int( file.split('_')[-1]) )  
        file_name_prev = file_name 

    for filename in sorted_list:
        filename_and_wav = filename + '.WAV'
        split_file_list_sorted.append(filename_and_wav)
    return split_file_list_sorted
     

def set_args(model_size='small', model_files_path='', output_path='', binary_prediction=False):
    # Start from default args
    args = default_args
    # Set constant paramteres
    args.dataset = 'birdcalls'
    args.dataset_mean = -0.22943932
    args.dataset_std = 0.93046004
    args.target_length = 128  # 1024 # 128
    args.noise = False
    args.bal = False
    # args.lr = 3e-4  # 3e-4
    # args.freqm=48 # 0
    # args.timem=48 # 0
    # args.mixup = 0.6  # 0.6
    args.fshape = 16
    args.tshape = 16
    args.fstride = 16  # 16
    args.tstride = 20  # 20  # 16  # Pavel change to 16 at Ron's it is 20
    args.task = 'ft_cls'
    # head_lr=1
    # args.warmup = False
    args.num_workers = 1
    args.pos_weight = torch.ones([args.n_class]).tolist()
    # args.batch_size = 128
    # args.n_epochs = 25
    args.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    # Configuration parameters
  

    args.pretrained_mdl_path = str(Path(model_files_path, 'audio_model_pretrained.pth'))
    if binary_prediction:
        args.fine_tuned_mdl_path = str(Path(model_files_path, 'audio_model_params_binary.pth'))    
        args.label_csv = str(Path(model_files_path, 'birdcalls_label_binary.csv'))    
    else:
        args.label_csv = str(Path(model_files_path, 'birdcalls_label.csv'))
        args.fine_tuned_mdl_path = str(Path(model_files_path, 'audio_model_params.pth'))
    
    label_df = pd.read_csv(args.label_csv)
    args.n_class = len(label_df)
    args.exp_dir = str(output_path)
    args.model_size = model_size
    # args.data_train = base_path + 'birdcalls_train_data_low_count_filter_10_u.json'
    # args.data_val = base_path + 'birdcalls_validate_data_low_count_filter_10_u.json'
    args.data_eval = str(Path(output_path, 'eval_wav_files.json'))

    return args