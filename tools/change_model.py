import sys
import copy
import torch
from prettytable import PrettyTable

if __name__ == '__main__':
    model_name = sys.argv[1]
    model = torch.load(model_name)
    new_model1 = copy.deepcopy(model)
    new_model2 = copy.deepcopy(model)
    table = PrettyTable(['original key', 'new key'])
    for key, value in model['state_dict'].items():
        if 'panoptic_head.transformer_decoder' in key:
            global_key = key.replace(
                'panoptic_head.transformer_decoder', 'panoptic_head.global_transformer_decoder')
            table.add_row([key, global_key])
            high2low_key = key.replace(
                'panoptic_head.transformer_decoder', 'panoptic_head.high2low_transformer_decoder')
            table.add_row([key, high2low_key])
            low2high_key = key.replace(
                'panoptic_head.transformer_decoder', 'panoptic_head.low2high_transformer_decoder')
            table.add_row([key, low2high_key])
            new_model1['state_dict'][high2low_key] = value
            new_model1['state_dict'][low2high_key] = value
            new_model2['state_dict'][global_key] = value
            new_model2['state_dict'][high2low_key] = value
            new_model2['state_dict'][low2high_key] = value
    table.align = 'l'
    print(table)
    save_path1 = model_name.replace('.pth', '_high_low.pth')
    save_path2 = model_name.replace('.pth', '_global_high_low.pth')
    print('Save to {}'.format(save_path1))
    print('Save to {}'.format(save_path2))
    torch.save(new_model1, save_path1)
    torch.save(new_model2, save_path2)
