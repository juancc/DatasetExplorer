"""
Explore classification dataset
Dataset forma: images inside folder/class

JCA
"""
import os
import argparse
from pathlib import Path


import matplotlib.pyplot as plt


from DatasetExplorer.analyzers import plot_mosaic, class_distribution, \
    sizes, complete_color_hist
from DatasetExplorer.auxfunc import create_out_dir

parser = argparse.ArgumentParser(
                    prog='DatasetExplorer',
                    description='Classification image dataset exploration')

parser.add_argument('path', help='Dataset location')


def main(path):
    print(f' -- DATASET EXPLORER --')

    # Create output folder. All the plots will be saved here
    out_path = create_out_dir(path, tag='exp')


    classes = os.listdir(path)
    print(f' - {len(classes)} in Dataset: {", ".join(classes)}')
    
    # Get all the files of the dataset
    files = list(Path(path).glob('**/*'))
    print(f' - {len(files)} Images on dataset')

    # Plot Mosaic
    # plot_mosaic(files)
    # plt.show()

    # Class distribution
    # class_distribution(files, classes)
    # plt.show()

    # Image sizez
    # sizes(files)
    # plt.show()

    # Image classes color histogram
    complete_color_hist(files, classes)
    # plt.show()
    plt.savefig("/Users/juanc/Downloads/archive/image_class_histogram.png")






if __name__ == '__main__':
    args = parser.parse_args()
    main(args.path)
