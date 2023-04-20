import os
import argparse
import pathlib
from os import path as osp

from lmdb_utils import scandir
from lmdb_utils import make_lmdb_from_imgs


def create_lmdb_for_div2k():
    """Create lmdb files for DIV2K dataset.
    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            * DIV2K_train_HR_sub
            * DIV2K_train_LR_bicubic/X2_sub
            * DIV2K_train_LR_bicubic/X3_sub
            * DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR images
    folder_path = os.path.join(pathlib.Path.home(), "datasets/image_restoration", "DIV2K/DIV2K_train_HR")
    lmdb_path = os.path.join(pathlib.Path.home(), "datasets/image_restoration", "DIV2K/DIV2K_train_HR.lmdb")
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def prepare_keys_div2k(folder_path):
    """Prepare image path list and keys for DIV2K dataset.
    Args:
        folder_path (str): Folder path.
    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        help=("Options: 'DIV2K', 'REDS', 'Vimeo90K' You may need to modify the corresponding configurations in codes."))
    args = parser.parse_args()
    dataset = args.dataset.lower()
    if dataset == 'div2k':
        create_lmdb_for_div2k()
    else:
        raise ValueError('Wrong dataset.')