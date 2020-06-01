
import glob
import os
from operator import add
import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (binary_dilation, binary_fill_holes,
                                      distance_transform_cdt,
                                      distance_transform_edt)
from skimage.morphology import remove_small_objects, watershed

import postproc.hover
import postproc.dist
import postproc.other

from config import Config

from misc.viz_utils import visualize_instances
from misc.utils import get_inst_centroid
from metrics.stats_utils import remap_label
import matplotlib.pyplot as plt

###################

# TODO: 
# * due to the need of running this multiple times, should make 
# * ioutputt less reliant on the training config file

## ! WARNING: 
## check the prediction channels, wrong ordering will break the code !
## the prediction channels ordering should match the ones produced in augs.py

cfg = Config()
cfg.nuclei_type_dict = {
            1:'Other',
            2:'Inflammatory',         
            3:'healthy Epithelial',   
            4:'malignant Epithelial', 
            5:'fibroblast',           
            6:'muscle',               
            7:'endothelial'
        }
cfg.nr_types = 8

# * flag for HoVer-Net only
# 1 - threshold, 2 - sobel based
energy_mode = 2 
marker_mode = 2 

pred_dir = cfg.inf_output_dir
proc_dir = pred_dir + '_proc'

file_list = glob.glob('%s/*.mat' % (pred_dir))
file_list.sort() # ensure same order

if not os.path.isdir(proc_dir):
    os.makedirs(proc_dir)

stat_output = [0] * (cfg.nr_types - 1)
for filename in file_list:
    filename = os.path.basename(filename)
    basename = filename.split('.')[0]
    print(pred_dir, basename, end=' ', flush=True)

    img = cv2.imread(cfg.inf_data_dir + basename + cfg.inf_imgs_ext)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    pred = sio.loadmat('%s/%s.mat' % (pred_dir, basename))
    pred_type = pred['type_map']
    pred_inst = pred['inst_map']
    print(pred_type.shape, pred_inst.shape, pred_inst.min(), pred_inst.max(), pred_type.min(), pred_type.max())
    # aqua, red, blue, green, orange, navy blue, yellow, medium gray
    clr_list = [[0, 128, 128], [255, 0, 0], [0, 0, 255], \
                [0, 255, 0], [255, 165, 0], [0, 0, 128], \
                [255, 255, 0], [128, 128, 128]]
    clr_list = clr_list[:cfg.nr_types - 1]

    overlaid_output, stat_out = visualize_instances(pred_inst, pred_type, canvas=img, color=clr_list, gt=True)
    stat_output = list(map(add, stat_out, stat_output))

    overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
    cv2.imwrite('%s/%s.png' % (proc_dir, basename), overlaid_output)

x_pos = list(range(cfg.nr_types - 1))
x_obj = [cfg.nuclei_type_dict[i + 1] for i in x_pos]
plt.bar(x_pos, stat_output, align='center', color=np.array(clr_list)/255.)
plt.xticks(x_pos, x_obj)
plt.ylabel('number of nucleus')
plt.show()

# break
print('FINISH')
