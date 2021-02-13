import numpy as np
from PIL import Image
import os

path = '/data/testdata_2/rgbd_dataset_freiburg3_long_office_household/depth'
path_out = '/data/DeepMSV_Colmap_solutions/Same_settings_ours/TUM_rgbd_dataset_freiburg3_long_office_household/gt'

tmp = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]

path_ass = '/data/testdata_2/rgbd_dataset_freiburg3_long_office_household/associate.txt'
f = open(path_ass, "r")
txt_file = dict()
for x in f:
    A = x.split(' ')
    txt_file[A[2]] = A[0]
 



for f in tmp:

    path_file = os.path.join(path, f)
    print(path_file)

    name = f.split('.')
    name = name[0] + '.' + name [1]
    
    try: 
        name_out = txt_file[name]
    except:
        print('not chenged :', name)
        name_out = name
    gt = Image.open(path_file)
    gt_depth = np.array(gt).astype(np.float32) / 5000

    np.save(os.path.join(path_out, name_out+'_rgb.png.depth.npy'), gt_depth)

