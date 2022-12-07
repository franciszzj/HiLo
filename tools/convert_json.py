import os
import sys
import json


def convert(in_file, out_file):
    fo = open(out_file, 'w')

    info_list = []
    with open(in_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            img_path = data['img_path']
            sub = data['sub']
            obj = data['obj']
            rel = data['rel']
            sub = change_name(sub)
            obj = change_name(obj)
            caption = '{} {} {}'.format(sub, rel, obj)
            info = {
                'image_path': img_path,
                'caption': caption
            }
            info_list.append(info)

    json.dump(info_list, fo)


def change_name(name):
    name = name.replace('-stuff', '')
    name = name.replace('-other', '')
    name = name.replace('-merged', '')
    return name


if __name__ == '__main__':
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    convert(in_file, out_file)
