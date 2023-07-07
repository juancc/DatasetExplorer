"""
Explore classification dataset
Dataset forma: images inside folder/class

JCA
"""
import os
import argparse
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from DatasetExplorer.analyzers import func_plotters
from DatasetExplorer.auxfunc import create_out_dir

parser = argparse.ArgumentParser(
                    prog='DatasetExplorer',
                    description='Classification image dataset exploration')

parser.add_argument('path', help='Dataset location')
parser.add_argument('-c', '--cache', help='Load in memory dataset. This could use all the machine RAM', action='store_true')


def main(path, cache=False, show=False):
    """ Analyze dataset located in path. Path is the direction to the root of 
    the dataset containing the folder by class.
    Args
    :param path: (str) path to root direcotry of the dataset
    :param cache: (bool) if true images of the dataset will be stored in memory
    :param show: (bool)show plots while performing analysis  
    """

    print(f' -- DATASET EXPLORER --')
    # Create output folder. All the plots will be saved here
    out_path = create_out_dir(path, tag='exp')

    classes = os.listdir(path)
    print(f' - {len(classes)} classes')
    
    # Get all the files of the dataset
    files = list(Path(path).glob('**/*'))
    print(f' - {len(files)} images')

    dataset = {
        'files': files,
        # Some analyzers load all the images. Keep them for speed analysis.
        # take into accound that this could use all the available RAM
        'images': {},
        'classes': classes,
    }


    if cache:
        print(f' - Loading images in memory...')
        err = []
        for im_path in tqdm(files, total=len(files)):
            try:
                im = cv2.imread(str(im_path))
            except Exception as e:
                err.append(im_path)
            else:
                 dataset['images'][im_path] = im


    init = time()

    print(' - Running: ')
    for func in func_plotters:
        func(dataset, show=show, save_path=out_path)
    print(f' -- Total time: {round(time()-init)} s --')






if __name__ == '__main__':
    args = parser.parse_args()
    main(args.path, cache=args.cache)
