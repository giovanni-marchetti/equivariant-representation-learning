import os
import urllib.request
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()

if not os.path.exists('data'):
    os.mkdir('data')

pbar = None
downloaded = 0

def show_progress(count, block_size, total_size):
    global pbar
    global downloaded
    if pbar is None:
        pbar = tqdm(total=total_size)

    downloaded += block_size
    pbar.update(block_size)
    if downloaded == total_size:
        pbar = None
        downloaded = 0

def download_platonic(dir='platonic'):
    global pbar
    global downloaded

    if not os.path.exists(f'data/{dir}'):
        os.mkdir(f'data/{dir}')

    base_url = 'https://storage.googleapis.com/equivariant-project/platonic'
    cube = 'cube_uniform_black-64-big.pkl'
    tetra = 'tetra_uniform_black-64-big.pkl'
    octa = 'octa_uniform_black-64-big.pkl'

    files = [cube, tetra, octa]
    print("Downloading platonics...")
    for f in files:
        urllib.request.urlretrieve(os.path.join(base_url, f), f'data/{dir}/{f}', show_progress)
        pbar = None
        downloaded = 0


def download_sprites_shapes(dir):
    global pbar
    global downloaded

    if not os.path.exists(f'data/{dir}'):
        os.mkdir(f'data/{dir}')

    base_url = f'https://storage.googleapis.com/equivariant-project/{dir}'
    files = ['equiv_classes.npy', 'equiv_data.npy', 'equiv_lbls.npy']

    for f in files:
        urllib.request.urlretrieve(os.path.join(base_url, f), f'data/{dir}/{f}', show_progress)
        pbar = None
        downloaded = 0


if __name__ == '__main__':
    if args.dataset == 'platonic':
        download_platonic()
    if args.dataset == 'sprites':
        download_sprites_shapes('sprites_data')
    if args.dataset == 'colorshift':
        download_sprites_shapes('colorshift_data')
    if args.dataset == 'multi-sprites':
        download_sprites_shapes('multisprites_data')
    




