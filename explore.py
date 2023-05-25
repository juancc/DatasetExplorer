"""
Explore classification dataset
Dataset forma: images inside folder/class

JCA
"""
import os
import argparse
from pathlib import Path


import matplotlib.pyplot as plt


from DatasetExplorer.analyzers import plot_mosaic


parser = argparse.ArgumentParser(
                    prog='DatasetExplorer',
                    description='Classification image dataset exploration')

parser.add_argument('path', help='Dataset location')


def main(path):
    classes = os.listdir(path)
    print(f' - {len(classes)} in Dataset: {", ".join(classes)}')
    
    # Get all the files of the dataset
    files = list(Path(path).glob('**/*'))
    print(f' - {len(files)} Images on dataset')

    # Plot Mosaic
    plot_mosaic(files)
    plt.show()





if __name__ == '__main__':
    args = parser.parse_args()
    main(args.path)