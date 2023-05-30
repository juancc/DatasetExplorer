"""
Some auxfunc for the analyzers

JCA
"""
import random

import cv2
import numpy as np



def automatic_contour(im, bck=2, convex_hull=True, **kwargs):
    """Contour is calculated automatic for images witout information
    Return object contour, area and centroid base background-object segmentation
        :param im: (np.array) image loaded for cv2
        :param bck: (int) background type (0:black, 1:white, 2:mixed)
        :convex_hull: (Bool) use convex hull for segmentation
    """
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Invert colors if clear (white) background
    if bck == 2: # Mixed
        # Infer background color by the median
        im_median = np.median(gray)
        bck = 0 if im_median < 255/2 else 1

    if bck == 1: # white
        gray = cv2.bitwise_not(gray)
    
    blur = cv2.GaussianBlur(gray,(3,3),0)
    ret, th = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull = None
    hull_area = 0
    hull_center = None
    # Only consider the larges convex hull of the image
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        if M['m00'] > hull_area:
            im_hull = cv2.convexHull(c) if convex_hull else c

            hull = im_hull 
            hull_area = M['m00']

            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            hull_center = (cX, cY)

    return hull, hull_area, hull_center



def calc_hist(im_path, label, cat_hist, pbar):
    """Calculate imagecolor histogram of an image
        : param im_path : (str) Image path
        : param label : (str) Image class label
        : param cat_hist: (dict) label: class_histogram
    """

    im = cv2.imread(str(im_path))

    try:
        im_hist = [ cv2.calcHist([im], [col_i], None, [256], [0, 256])\
                    for col_i, col in enumerate(['b', 'g', 'r']) ]
        im_hist = np.vstack(im_hist)

        # divide per image area
        h, w = im.shape[:2]
        area = h*w
        im_hist /= area

        # Normalize
        im_hist /= max(im_hist)

    except Exception as e:
        print(f'Err {e} reading image: {im_path}')
    else:
        if label in cat_hist:
            cat_hist[label] += im_hist
        else:
            cat_hist[label] = im_hist
    
    pbar.update()