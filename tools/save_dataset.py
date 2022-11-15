import os
import sys
import time
from mmcv import Config, DictAction
from openpsg.datasets import build_dataset


if __name__ == '__main__':
    config = sys.argv[1]
    cfg = Config.fromfile(config)

    dataset = build_dataset(cfg.data.train)
    # dataset = build_dataset(cfg.data.test)
    start = time.time()
    for i, d in enumerate(dataset):
        if i % 1000 == 0:
            end = time.time()
            print('{}/{}, samples/sec: {} ...'.format(i,
                  len(dataset), (i+1)/(end-start)))
