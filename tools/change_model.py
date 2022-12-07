import sys
import copy
import torch

if __name__ == '__main__':
    model_name = sys.argv[1]
    model = torch.load(model_name)
    new_model = copy.deepcopy(model)
    for key, value in model['state_dict'].items():
        if 'panoptic_head.transformer_decoder' in key:
            high2low_key = key.replace(
                'panoptic_head.transformer_decoder', 'panoptic_head.high2low_transformer_decoder')
            low2high_key = key.replace(
                'panoptic_head.transformer_decoder', 'panoptic_head.low2high_transformer_decoder')
            new_model['state_dict'][high2low_key] = value
            new_model['state_dict'][low2high_key] = value
    torch.save(new_model, model_name.replace('.pth', '_high_low.pth'))
