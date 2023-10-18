import os
import numpy as np
from pathlib import Path
import pandas as pd


def read_optical_motion():
    path = '/home/kapilchoudhary/Downloads/DKnerf/data/dynerf/Sparse_flow'
    files = Path(os.path.join(path, 'sparse_flow_csvs')).glob('*.csv')
    f1c1_f2c2 = []

    # missing_opt_flow_file = Path(path).glob('*.txt')
    # dict1 = {}
    # dict2 = {}

    # for j, k in enumerate(missing_opt_flow_file):
    #     missed = pd.read_csv(k, header=None)
    #     missed = missed[0]

    dfs = list()
    for i, f in enumerate(files):
        f1c1_f2c2.append(f.stem)
        data = pd.read_csv(f)
        # dict1['camera'] = f'{f.stem}'
        dfs.append(data)

    return dfs, f1c1_f2c2


#read_optical_motion()
#  --log-dir logs/realdynamic/cutbeef_explicit/adding_3d_motion_consistancy/correct_l1_time_loss/3d_motion_loss_with_lr=.001 --validate-only