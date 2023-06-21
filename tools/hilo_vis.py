import os
import sys
import cv2
import json
import copy
import numpy as np
import mmcv
from skimage.segmentation import find_boundaries
from panopticapi.utils import rgb2id, id2rgb
from prettytable import PrettyTable

file_client = mmcv.FileClient(**dict(backend='disk'))


PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
           (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
           (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
           (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
           (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
           (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
           (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
           (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
           (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
           (134, 134, 103), (145, 148, 174), (255, 208, 186),
           (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
           (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
           (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
           (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
           (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
           (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
           (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
           (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
           (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
           (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
           (191, 162, 208), (255, 255, 128), (147, 211, 203),
           (150, 100, 100), (168, 171, 172), (146, 112, 198),
           (210, 170, 100), (92, 136, 89), (218, 88, 184), (241, 129, 0),
           (217, 17, 255), (124, 74, 181), (70, 70, 70), (255, 228, 255),
           (154, 208, 0), (193, 0, 92), (76, 91, 113), (255, 180, 195),
           (106, 154, 176),
           (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
           (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
           (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
           (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
           (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
           (146, 139, 141),
           (70, 130, 180), (134, 199, 156), (209, 226, 140), (96, 36, 108),
           (96, 96, 96), (64, 170, 64), (152, 251, 152), (208, 229, 228),
           (206, 186, 171), (152, 161, 64), (116, 112, 0), (0, 114, 143),
           (102, 102, 156), (250, 141, 255)]


def t(color, up=100):
    return (min(color[0] + up, 255), min(color[1] + up, 255), min(color[2] + up, 255))


def vis(img_prefix, img_file, seg_file, meta_info, object_classes, predicate_classes):
    img = cv2.imread(img_file)
    seg_bytes = file_client.get(seg_file)
    seg = mmcv.imfrombytes(seg_bytes,
                           flag='color',
                           channel_order='rgb').squeeze()
    seg_id = rgb2id(seg)

    boundaries = find_boundaries(seg_id, mode='thick')

    new_seg = copy.deepcopy(img)
    new_mask = np.zeros_like(img)
    for idx, object in enumerate(meta_info['segments_info']):
        object_id = object['id']
        object_label = object['category_id']
        index = np.where(seg_id == object_id)
        # new_seg[index] = PALETTE[object_label]
        if idx == 4:
            new_seg[index] = [128, 128, 128]
        else:
            new_seg[index] = PALETTE[idx * 2 % len(PALETTE)]
        if idx in [0, 2]:
            new_mask[index] = [255, 255, 255]

    cv2.imwrite('./{}_mask_0_2.png'.format(img_prefix), new_mask)

    vis_img = img * 0.5 + new_seg * 0.5
    vis_img[boundaries] = [0, 0, 0]

    name_list = []
    pos_list = []
    rgb_list = []
    for idx, object in enumerate(meta_info['segments_info']):
        object_id = object['id']
        object_label = object['category_id']
        object_name = object_classes[object_label]
        y_idx, x_idx = np.where(seg_id == object_id)
        # rgb = id2rgb(object_id)
        # rgb = PALETTE[object_label]
        if idx == 4:
            rgb = [128, 128, 128]
        else:
            rgb = PALETTE[idx * 2 % len(PALETTE)]
        y = int(y_idx.mean())
        x = int(x_idx.mean())
        name = '{}_{}'.format(idx, object_name)
        name_list.append(name)
        pos_list.append((x, y))
        rgb_list.append(rgb)

    for name, pos, rgb in zip(name_list, pos_list, rgb_list):
        x = pos[0]
        y = pos[1]
        # cv2.putText(vis_img, name, (x - len(name), y + 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, thickness=1)

    cv2.imwrite('./{}_vis.jpg'.format(img_prefix), vis_img)

    table = PrettyTable(['subject', 'object', 'relation'])
    for relation in meta_info['relations']:
        table.add_row([name_list[relation[0]],
                      name_list[relation[1]], predicate_classes[relation[2]]])
    print(table)


def process(img_prefix, img_root, seg_root, psg_file):
    img_file = os.path.join(img_root, img_prefix+'.jpg')
    seg_file = os.path.join(seg_root, img_prefix+'.png')
    psg = json.load(open(psg_file))
    data = psg['data']
    thing_classes = psg['thing_classes']
    stuff_classes = psg['stuff_classes']
    predicate_classes = psg['predicate_classes']
    object_classes = thing_classes + stuff_classes
    for d in data:
        if img_prefix in d['file_name']:
            vis(img_prefix, img_file, seg_file, d,
                object_classes, predicate_classes)


if __name__ == '__main__':
    img_prefix = sys.argv[1]
    img_root = '/jmain02/home/J2AD019/exk01/zxz35-exk01/workspace/OpenPSG/data/coco/train2017'
    seg_root = '/jmain02/home/J2AD019/exk01/zxz35-exk01/workspace/OpenPSG/data/coco/panoptic_train2017'
    psg_file = '/jmain02/home/J2AD019/exk01/zxz35-exk01/workspace/OpenPSG/data/psg/psg.json'

    process(img_prefix, img_root, seg_root, psg_file)
