import os
import sys
import json


def select_best_result(log_json_file):
    result_list = []
    with open(log_json_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'sgdet_recall_R_20' in data.keys() and \
                    data['mode'] == 'val':
                R_20 = float(data['sgdet_recall_R_20'])
                mR_20 = float(data['sgdet_mean_recall_mR_20'])
                epoch = int(data['epoch'])
                result_list.append((epoch, R_20, mR_20, data))

    R_weight = 1/3
    mR_weight = 2/3
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


if __name__ == '__main__':
    in_file = sys.argv[1]
    select_best_result(in_file)
