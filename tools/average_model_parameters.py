import os
import sys
import copy
import torch


def merge_model(model1_path, model2_path, save_path):
    model1 = torch.load(model1_path)
    model2 = torch.load(model2_path)
    model = copy.deepcopy(model1)
    for key in model1['state_dict'].keys():
        if 'backbone' in key or 'bbox_head.pixel_decoder' in key:
            value = (model1['state_dict'][key] + model2['state_dict'][key])/2
        else:
            value = model1['state_dict'][key]
        model['state_dict'][key] = value
    torch.save(model, save_path)


if __name__ == '__main__':
    model1_path = sys.argv[1]
    model2_path = sys.argv[2]
    save_path = sys.argv[3]

    merge_model(model1_path, model2_path, save_path)
