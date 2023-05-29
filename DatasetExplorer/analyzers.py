"""
Functions to analyze each image of the dataset

JCA
"""
import os
import random

import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

from DatasetExplorer.auxfunc import automatic_contour

def plot_mosaic(files, size=(3,3)):
    """Get mosaic plot
        :param files: list of all images on dataset
        :param size: (Tuple) Ros and columns of the dataset
    """
    cat_im_path = random.choices(files, k=size[0]*size[1])
    i=1
    for im_path in cat_im_path:
        plt.subplot(size[0], size[1], i)
        plt.subplots_adjust(top=1)
        plt.axis('off')
        plt.tight_layout()

        class_name = str(im_path).split(os.sep)[-2]
        plt.title(class_name, fontsize=10)

        try:
            im = cv2.imread(str(im_path))
            # hull, hull_area, hull_center = automatic_contour(im, convex_hull=True)
            # cv2.drawContours(im, [hull], -1, (255))

            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f'Err {e} reading image: {im_path}')
        
        i += 1

def class_distribution(files, classes, color='dodgerblue', size=(16, 9)):
    """Get classes distribution
        :param files: (list) of all images on dataset
        :param classes: (List) of the names of the dataset classes
    """
    clss = [str(f).split(os.sep)[-2] for f in files]
    classes_count = [clss.count(c) for c in classes]

    fig, ax = plt.subplots(figsize=size)
    plt.barh(classes, classes_count, color=color)

    # Add bar count number
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.2,
                str(round((i.get_width()), 2)),
                fontsize = 10)

    # Average files
    av = np.mean(classes_count)
    plt.axvline(x=av, color = 'grey', linestyle='dashed')


def sizes(files)->None:
    """Display the distribution of image shapes of the dataset.
          - x-axis: width
          - y-axis: height
          - diameter: observation count
        :param files: (list) of all images on dataset
    """
    obs_size = {} # (x,y): count
    # n = 0
    with tqdm(total=len(files)) as pbar:
        for im_path in files:
            try:
                im = cv2.imread(str(im_path))
                h, w = im.shape[:2]
            except Exception as e:
                print(f'Err {e} reading image: {im_path}')
            else:
                if (w,h) in obs_size:
                    obs_size[(w,h)] += 1
                else:
                    obs_size[(w,h)] = 1
                pbar.update()

    x = []
    y = []
    scale = []
    for k,v in obs_size.items():
        x.append(k[0])
        y.append(k[1])
        scale.append(v)

    plt.figure(figsize=(15,15))
    plt.scatter(x,y, s=scale, alpha=0.4, edgecolor='black', c='blue')
    max_size = max(x+y) * 1.1
    plt.plot([0,max_size], [0, max_size], '--', color='black')


def calc_hist(ims_path, label, cat_color, pbar):
    """Calculate imagecolor histogram of an image"""

    im = cv2.imread(str(ims_path))

    try:
        im_hist = [ cv2.calcHist([im], [col_i], None, [256], [0, 256]) for col_i, col in enumerate(['b', 'g', 'r']) ]
        im_hist = np.vstack(im_hist)

        # divide per image area
        h, w = im.shape[:2]
        area = h*w
        im_hist /= area

        # Normalize
        im_hist /= max(im_hist)

    except Exception as e:
        print(f'Err {e} reading image: {ims_path}')
    else:
        if label in cat_color:
            cat_color[label] += im_hist
        else:
            cat_color[label] = im_hist
    
    pbar.update()
        
def complete_color_hist(files, classes, color_alpha=0.5)->None:
    """
        Display the color distribution per class. 
        The histogram of percentage of pixel count per color of all the images of a class
        Histogram of pixel count / image size per color

        WARNING: This class loads all the dataset images if not loaded before

    """
    # pixel count per RGB (0-256) per chanel for all the images per category
    # R: [0:256]
    # B: [256:512]
    # G: [512:768]
    cat_color = {} # {id: [number of pixels per color]}
    with tqdm(total=len(files)) as pbar:
        for ims_path in files:
            label = str(ims_path).split(os.sep)[-2] 
            calc_hist(ims_path, label, cat_color, pbar)

    # n = len([ idx for idx, cat in dataset.labels.items() if idx in cat_color])
    n = len(classes)
    fig, axs = plt.subplots(n, constrained_layout = True)
    fig.set_figheight(n*2.5)

    red = (1,0,0, color_alpha)
    green = (0,1,0, color_alpha)
    blue = (0,0,1, color_alpha)

    fig_ax = 0
    for idx in classes:
        if idx in cat_color:
            axs[fig_ax].set_title(idx)
            for i in range(0, 256):
                axs[fig_ax].bar(i, cat_color[idx][i+512], color=red, linewidth=0, width=1.0)
                axs[fig_ax].bar(i, cat_color[idx][i], color=blue, linewidth=0, width=1.0)
                axs[fig_ax].bar(i, cat_color[idx][i+256], color=green, linewidth=0, width=1.0) 
            fig_ax += 1   