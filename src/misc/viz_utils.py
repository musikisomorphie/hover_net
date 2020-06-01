
import cv2
import math
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from .utils import bounding_box
from collections import Counter

####
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

####
def visualize_instances(mask, mask_type=None, canvas=None, color=None, gt=False):
    """
    Args:
        mask: array of NW
    Return:
        Image with the instance overlaid
    """
    canvas = np.full(mask.shape + (3,), 200, dtype=np.uint8) \
                if canvas is None else np.copy(canvas)

    insts_list = list(np.unique(mask))
    insts_list.remove(0) # remove background
    inst_colors = random_colors(len(insts_list))
    inst_colors = np.array(inst_colors) * 255

    inst_stat = [0] * len(color)
    for idx, inst_id in enumerate(insts_list):
        if mask_type is not None:
            assert color is not None, 'color can not be None when mask_type is given.'
            clr_idx = mask_type[mask == inst_id]
            # if compute the predicted nuclei color, 
            #    then compute the bins of different colors and use the maximum one
            # else compute the groundtruth nuclei color
            #    which is identical for the entire nuclei thus take clr_idx[0]
            clr_nu = np.argmax(np.bincount(clr_idx)) - 1 if not gt \
                     else int(clr_idx[0])
            clr_nu -= 1
            inst_color = color[clr_nu]
            inst_stat[clr_nu] += 1
        else:
            inst_color = color[idx] if color is not None else inst_colors[idx]
        inst_map = np.array(mask == inst_id, np.uint8)
        y1, y2, x1, x2  = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1 
        x1 = x1 - 2 if x1 - 2 >= 0 else x1 
        x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2 
        y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2 
        inst_map_crop = inst_map[y1:y2, x1:x2] if not gt else cv2.UMat(inst_map[y1:y2, x1:x2])
        inst_canvas_crop = canvas[y1:y2, x1:x2]
        contours = cv2.findContours(inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(inst_canvas_crop, contours[1], -1, inst_color, 2)
        canvas[y1:y2, x1:x2] = inst_canvas_crop        
    return canvas, inst_stat

####
def gen_figure(imgs_list, titles, fig_inch, shape=None,
                share_ax='all', show=False, colormap=plt.get_cmap('jet')):

    num_img = len(imgs_list)
    if shape is None:
        ncols = math.ceil(math.sqrt(num_img))
        nrows = math.ceil(num_img / ncols)
    else:
        nrows, ncols = shape

    # generate figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                        sharex=share_ax, sharey=share_ax)
    axes = [axes] if nrows == 1 else axes

    # not very elegant
    idx = 0
    for ax in axes:
        for cell in ax:
            cell.set_title(titles[idx])
            cell.imshow(imgs_list[idx], cmap=colormap)
            cell.tick_params(axis='both', 
                            which='both', 
                            bottom='off', 
                            top='off', 
                            labelbottom='off', 
                            right='off', 
                            left='off', 
                            labelleft='off')
            idx += 1
            if idx == len(titles):
                break
        if idx == len(titles):
            break
 
    fig.tight_layout()
    return fig
####
