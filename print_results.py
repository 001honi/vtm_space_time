from datetime import datetime
import numpy as np
import torch
import pandas as pd
import os
import argparse
from functools import cmp_to_key
from tabulate import tabulate


DATASETS = ['davis2016', 'davis2017', 'isic2018', 'duts', 'ap10k', 'eigen']
TASKS = ['vos', 'mvos', 'ds', 'sod', 'animalkp', 'depth']
MODELS = {
    0: ['VTM_unified_LARGE_noCont_noCat', 
        'VTM_unified_LARGE_benchmark_table5_2_base_pseudo', 
        'VTM_unified_LARGE_noCat', 'VTM_unified_LARGE_noCont', 
        'VTM_unified_LARGE', 
        'VTM_unified_LARGE_benchmark', 
        'VTM_unified_LARGEtable_5_2_segonly', 
        'VTM_unified_LARGE_table5_2_noseg'],
    1: ['VTM_unified_LARGE_benchmark_taskonomy_only',
        'VTM_unified_LARGE_benchmark_table5_3_midair_pseudo',
        'VTM_unified_LARGE_benchmark_target_domain_pseudo_except_taskonomy',
        'VTM_unified_LARGE_benchmark_table5_3_alldata_uniform'],
    2: [
        'VTM_unified_LARGE_table5_4_nobf',
        'VTM_unified_LARGE_benchmark_table5_4_l2b10'],
    3: [
        'VTM_unified_LARGE_noCont_noCat',
        'VTM_unified_LARGE_benchmark_table5_2_base_pseudo',
        'VTM_unified_LARGE_noCat',
        'VTM_unified_LARGE',
        'VTM_unified_LARGE_benchmark',
    ],
    4: [
        'VTM_task:all',
        'VTM_task:all_LARGE',
        'VTM_task:all_LARGE_nl:8_mc:4',
    ],
    5: [
        'VTM_unified_LARGE_benchmark'
    ],
    6: [
        'VTM_unified_LARGE_benchmark_without_l2b',
        'VTM_unified_LARGE_benchmark_table5_4_l2b10',
        'VTM_unified_LARGE_benchmark',
    ]
}
   
dataset_dict = {
    'davis2016': {
        'vos': ('Video Object Segmentation', 'J&F-Mean (↑)'),
    },
    'davis2017': {
        'vos': ('Video (Multiple) Object Segmentation', 'J&F-Mean (↑)'),
    },
    'isic2018': {
        'ds': ('Dermoscopic Image Segmentation', 'DICE (↑)'),
    },
    'duts': {
        'sod': ('Salient Object Segmentation', 'Max.F (↑)'),
    },
    'ap10k': {
        'animalkp': ('Animal Keypoint', 'AP (↑)'),
    },
    'eigen': {
        'depth': ('Depth Prediction', 'Abs.Rel (↓)'),
    }
}

shots_dict = {
    'davis2016': [1, 2, 3, 4],
    'davis2017': [1],
    'isic2018': [1, 2, 5, 10],
    'duts': [1, 2, 5, 10],
    'ap10k': [1, 5, 10, 20, 50],
    'eigen': [1, 5, 10, 20, 40, 50]
}
shots_dict = {k: [str(i) for i in v] for k, v in shots_dict.items()}

def safe_listdir(path):
    try:
        return os.listdir(path)
    except:
        return []


def update_results(table, row, col, value, tag, mode='max'):
    if row not in table.index:
        table = pd.concat([table, pd.DataFrame(index=[row], columns=table.columns)])
    if mode == 'max':
        if not isinstance(table.loc[row][col], tuple):
            table.loc[row][col] = (f'{value:.04f}', 1, tag)

        else:
            old_value, count, old_tag = table.loc[row][col]
            if value > float(old_value):
                table.loc[row][col] = (f'{value:.04f}', count + 1, tag)
            else:
                table.loc[row][col] = (old_value, count + 1, old_tag)
    else:
        if not isinstance(table.loc[row][col], tuple):
            table.loc[row][col] = (f'{value:.04f}', 1, tag)
        else:
            old_value, count, old_tag = table.loc[row][col]
            if value < float(old_value):
                table.loc[row][col] = (f'{value:.04f}', count + 1, tag)
            else:
                table.loc[row][col] = (old_value, count + 1, old_tag)
    return table


def add_results(table, row, col, value, tag, mode='max'):
    if row not in table.index:
        table = pd.concat([table, pd.DataFrame(index=[row], columns=table.columns)])
    if not isinstance(table.loc[row][col], dict):
        table.loc[row][col] = {tag:[value], 'mode':mode}

    else:
        if tag in table.loc[row][col]:
            table.loc[row][col][tag].append(value)
        else:
            table.loc[row][col][tag] = [value]

    return table


def process_dups(table):
    for row in table.index:
        for col in table.columns:
            if isinstance(table.loc[row][col], dict):
                res_dict = table.loc[row][col]
                mode = res_dict.pop('mode')
                aggregate = {}
                for tag in res_dict:
                    aggregate[tag] = (np.mean(res_dict[tag]), np.std(res_dict[tag]), len(res_dict[tag]))
                if mode == 'max':
                    best_tag = max(aggregate, key=lambda x: aggregate[x][0])
                else:
                    best_tag = min(aggregate, key=lambda x: aggregate[x][0])
                all_results = list(res_dict.values())[0]
                all_results = [f'{x:.04f}' for x in all_results]
                table.loc[row][col] = (f'{aggregate[best_tag][0]:.04f} \u00B1 {aggregate[best_tag][1]:.04f}' , f'{aggregate[best_tag][2]}', best_tag, all_results)


def create_database(task_dict, exp_name=None, subname_prefix=None, print_failure=False, from_last=False, compact=False):
    result_root = os.path.join('experiments', args.result_dir)
    
    # create indices with model names
    if exp_name is not None:
        model_names = [exp_name]
    else:
        model_names = sorted(safe_listdir(result_dir))
        if args.table_num is not None:
            model_names = MODELS[args.table_num]
        if args.exp_pattern:
            model_names = [model_name for model_name in model_names if (args.exp_pattern in model_name)]
    
    # create multi-columns with dataset, task, and metric
    keys = []
    for dataset in datasets:
        for shot in shots_dict[dataset]:
            keys += [(dataset.upper(), shot, task_name, metric, dataset, task)
                    for task, (task_name, metric) in task_dict[dataset]]
            if compact:
                break
    columns = [key[:4] for key in keys]
    columns = pd.MultiIndex.from_tuples(columns, names=['Dataset', 'Shot', 'Task', 'Metric'])

    result_ptf = '_fromlast' if from_last else ''
    
    # construct a database
    database = pd.DataFrame(index=model_names, columns=columns)
    for model_name in model_names:
        for (*column, dataset, task) in keys:
            column = tuple(column)
            exp_name = model_name
            exp_dir = os.path.join(result_root, exp_name)
            if not os.path.exists(exp_dir):
                continue

            shot = column[1]
            if dataset == 'davis2016':
                result_name = f'davis2016_vos_results_shot:{shot}{result_ptf}/global_results-val.csv'
            elif dataset == 'davis2017':
                result_name = f'davis2017_mvos_results_shot:{shot}{result_ptf}/global_results-val.csv'
            elif dataset == 'isic2018':
                result_name = f'isic2018_ds_results_shot:{shot}{result_ptf}.pth'
            elif dataset == 'duts':
                result_name = f'duts_sod_results_shot:{shot}{result_ptf}.pth'
            elif dataset == 'ap10k':
                result_name = f'ap10k_animalkp_results_shot:{shot}{result_ptf}/result.pth'
            elif dataset == 'eigen':
                result_name = f'{dataset}_depth_results_shot:{shot}.pth'
            else:
                raise NotImplementedError
            

            subname_ptf_list = list(set(['_'.join(exp_subname.split('_')[1:]) for exp_subname in safe_listdir(exp_dir)]))
            for subname_ptf in subname_ptf_list:
                exp_subnames = [exp_subname for exp_subname in safe_listdir(exp_dir)
                                if '_'.join(exp_subname.split('_')[1:]) == subname_ptf]
                if subname_prefix is not None:
                    exp_subnames = [exp_subname for exp_subname in exp_subnames if all(subname_prefix[i] in exp_subname for i in range(len(subname_prefix)))]
                
                if '_shot:' in subname_ptf:
                    vis_subname_ptf = subname_ptf.replace(f'_shot:{shot}', '')
                else:
                    vis_subname_ptf = subname_ptf

                if '_support_idx' in vis_subname_ptf:
                    for i in range(1, 6):
                        vis_subname_ptf = vis_subname_ptf.replace(f'_support_idx:{i}', '')
                vis_subname_ptf = vis_subname_ptf.replace('_skip_crowd:True_ssl:True_top_one:True', '')

                row = f'{model_name} ({vis_subname_ptf})' if subname_ptf != '' else model_name
                for exp_subname in exp_subnames:
                    result_path = os.path.join(exp_dir, exp_subname, 'logs', result_name)
                    if os.path.exists(result_path):
                        mode = 'max'
                        if result_path.endswith('.csv'):
                            result = pd.read_csv(result_path)
                        else:
                            result = torch.load(result_path)
                        if dataset in ['davis2016', 'davis2017']:
                            value = result['J&F-Mean'][0]
                        elif dataset == 'linemod':
                            value = result[2]
                        elif dataset == 'isic2018':
                            value = result
                        elif dataset == 'duts':
                            value = result['Max.F']
                        elif dataset == 'ap10k':
                            value = result[0][1]
                        elif dataset == 'eigen':
                            value = result['abs_rel']
                            mode = 'min'
                        tag = exp_subname.replace(f'_{subname_ptf}', '')
                        if args.lr is None or tag == f'lr:{args.lr}':
                            database = add_results(database, row, column, value, tag, mode)
                    elif print_failure:
                        print(result_path)

    process_dups(database)
    return database


if True:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='+', default=None, choices=DATASETS)
    parser.add_argument('--result_dir', type=str, default='TEST')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--exp_pattern', type=str, default=None)
    parser.add_argument('--table_num', type=int, default=None)
    parser.add_argument('--subname_prefix', '-sprf', type=str, default=None, nargs='+')
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    parser.add_argument('--from_last', '-fl', default=False, action='store_true')
    parser.add_argument('--compact', '-cp', default=False, action='store_true')
    parser.add_argument('--lr', type=str, default=None)
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()
    

    result_dir = os.path.join('experiments', args.result_dir)

    # choose datasets to show
    if args.dataset is not None:
        datasets = args.dataset
    else:
        datasets = list(set([
            log.split('_')[0]
            for exp_name in safe_listdir(result_dir)
            for exp_subname in safe_listdir(os.path.join(result_dir, exp_name))
            for log in safe_listdir(os.path.join(result_dir, exp_name, exp_subname, 'logs'))
        ]))
        datasets = [dataset.replace('result', 'taskonomy') for dataset in datasets]
        datasets = [dataset for dataset in datasets if dataset in DATASETS]

    # construct a task dictionary
    task_dict = {}
    for dataset in datasets:
        task_dict[dataset] = []
        for task in TASKS:
            if dataset in dataset_dict and task in dataset_dict[dataset]:
                task_dict[dataset].append((task, dataset_dict[dataset][task]))

    try:
        pd.set_option('max_columns', None)
    except:
        pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.colheader_justify', 'left')
    database = create_database(task_dict, args.exp_name,
                               args.subname_prefix, print_failure=args.verbose, from_last=args.from_last,
                               compact=args.compact)
    database = database[database.columns[database.isna().sum(axis=0) < len(database.index)]]
    database = database.loc[database.index[database.isna().sum(axis=1) < len(database.columns)]]
    database = database.reindex(sorted(database.columns,
                                       key=cmp_to_key(lambda x, y: DATASETS.index(x[0].lower()) - 
                                                                   DATASETS.index(y[0].lower()))), axis=1)
    
    print(database.to_string(justify='right'))
    if args.save:
        # current time
        database.to_csv(f'results_{datetime.now().strftime("%Y%m%d_%H%M%S") }.csv')
