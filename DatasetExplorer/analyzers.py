"""
Functions to analyze each image of the dataset

JCA
"""
import os
import random

import matplotlib.pyplot as plt
import cv2

from DatasetExplorer.auxfunc import automatic_contour

def plot_mosaic(filenames, size=16):
    """Plot mosaic
        :param filenames: list of all images on dataset
        :param size: (int) number of images to plot
    """
    # List dir -> check is a file and an image, 
    # choice a random path -> load image with PIL
    cat_im_path = random.choices(filenames, k=size)
    is_title = False

    row_col = int(size/4)

    i=1
    for im_path in cat_im_path:
        plt.subplot(row_col, row_col, i)
        plt.subplots_adjust(top=1)
        plt.axis('off')
        plt.tight_layout()

        class_name = str(im_path).split(os.sep)[-2]
        plt.title(class_name, fontsize=10)

        try:
            im = cv2.imread(str(im_path))
            hull, hull_area, hull_center = automatic_contour(im)
            cv2.drawContours(im, [hull], -1, (255))


            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f'Err {e} reading image: {im_path}')
        
        i += 1