import os
import sys
import json
import numpy as np


def select_best_result(log_json_file, select_type):
    result_list = []
    with open(log_json_file, 'r') as f:
        for line in f:
            if 'val' not in line:
                continue
            data = json.loads(line.strip())
            if 'sgdet_recall_R_20' in data.keys() and \
                    'mode' in data.keys() and \
                    data['mode'] == 'val':
                R_20 = float(data['sgdet_recall_R_20'])
                mR_20 = float(data['sgdet_mean_recall_mR_20'])
                epoch = int(data['epoch'])
                result_list.append((epoch, R_20, mR_20, data))

    if select_type == 'all':
        R_weight = 1/3
        mR_weight = 2/3
    elif select_type == 'R':
        R_weight = 1
        mR_weight = 0
    elif select_type == 'mR':
        R_weight = 0
        mR_weight = 1
    best_result_info = (0, 0, 0, '')
    for result in result_list:
        best_result = best_result_info[1] * R_weight + \
            best_result_info[2] * mR_weight
        current_result = result[1] * R_weight + result[2] * mR_weight
        if current_result > best_result:
            best_result_info = result

    print('##### Best Result for {} #####'.format(log_json_file))
    print('epoch: {}'.format(best_result_info[0]))
    print('R_20: {:.2f}'.format(best_result_info[1] * 100))
    print('mR_20: {:.2f}'.format(best_result_info[2] * 100))
    print(best_result_info[3])


def select_best_epoch(log_json_file):
    epoch_list = []
    loss_list = []
    with open(log_json_file, 'r') as f:
        for line in f:
            if 'val' not in line:
                continue
            data = json.loads(line.strip())
            if 'loss' in data.keys() and \
                    'mode' in data.keys() and \
                    data['mode'] == 'val':
                epoch = data['epoch']
                loss = data['loss']
                epoch_list.append(epoch)
                loss_list.append(loss)

    if len(epoch_list) == 0:
        print('epoch zero bug.')
        return

    idx = np.argmin(np.array(loss_list))
    best_epoch = epoch_list[idx]

    print('##### Best Result for {} #####'.format(log_json_file))
    print('epoch: {}'.format(best_epoch))
    print('loss: {:.2f}'.format(loss_list[idx]))


if __name__ == '__main__':
    in_file = sys.argv[1]
    try:
        select_type = sys.argv[2]
    except:
        select_type = 'all'

    if select_type == 'loss':
        select_best_epoch(in_file)
    else:
        select_best_result(in_file, select_type)
