import gzip
import os
import shutil

import gdown
import requests
from tqdm import tqdm

from src import RAW_DIR, ExperimentConfig, MODEL_DIR


def dl_visual_feature(chunk_size: int = 10000):

    fname = os.path.join(RAW_DIR, "image_features_tradesy.b")
    if not os.path.isfile(fname):
        resp = requests.get(r"http://jmcauley.ucsd.edu/data/tradesy/image_features_tradesy.b", stream=True)

        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(desc="Downloading binary file containing visual features...",
                                             total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:

            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

        print(f"\nVisual features downloaded into {fname}!")
    else:
        print(f"Visual features were already downloaded into {fname}, skipped")


def dl_tradesy_feedback():

    fname = os.path.join(RAW_DIR, "tradesy.json")
    if not os.path.isfile(fname):
        gdown.download("https://drive.google.com/uc?id=1xaRS4qqGeTzxaEksHzjVKjQ6l7QT9eMJ",
                       output=os.path.join(RAW_DIR, "tradesy.json.gz"))

        with gzip.open(os.path.join(RAW_DIR, "tradesy.json.gz"), 'rb') as f_in:
            with open(fname, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(os.path.join(RAW_DIR, "tradesy.json.gz"))

        print(f"\nTradesy raw feedback downloaded into {fname}!")
    else:
        print(f"Tradesy raw feedback were already downloaded into {fname}, skipped")


def dl_extended_tradesy_images(chunk_size: int = 10000):

    fname = os.path.join(RAW_DIR, "TradesyImgPartitioned.npy")
    if not os.path.isfile(fname):
        resp = requests.get(r"http://cseweb.ucsd.edu/~wckang/DVBPR/TradesyImgPartitioned.npy", stream=True)

        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(desc="Downloading npy matrix containing tradesy images...",
                                             total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:

            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

        print(f"\nTradesy images downloaded into {fname}!")
    else:
        print(f"Tradesy raw images were already downloaded into {fname}, skipped")


def dl_caffe_files(chunk_size: int = 10000):

    caffe_model_dir = os.path.join(MODEL_DIR, "reference_caffenet")
    os.makedirs(caffe_model_dir, exist_ok=True)

    mean_fname = os.path.join(caffe_model_dir, "ilsvrc_2012_mean.npy")

    if not os.path.isfile(mean_fname):
        resp = requests.get(r"https://github.com/facebookarchive/models/raw/master/bvlc_reference_caffenet/ilsvrc_2012_mean.npy", stream=True)

        total = int(resp.headers.get('content-length', 0))
        with open(mean_fname, 'wb') as file, tqdm(desc="Downloading mean ImageNet pixel file...",
                                             total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:

            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

        print(f"\nMean ImageNet pixel file downloaded into {mean_fname}!")
    else:
        print(f"Mean ImageNet pixel file was already downloaded into {mean_fname}, skipped")

    model_fname = os.path.join(caffe_model_dir, "bvlc_reference_caffenet.caffemodel")

    if not os.path.isfile(model_fname):
        resp = requests.get(r"http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel", stream=True)

        total = int(resp.headers.get('content-length', 0))
        with open(model_fname, 'wb') as file, tqdm(desc="Downloading Caffe reference model...",
                                             total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:

            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

        print(f"\nCaffe reference model downloaded into {model_fname}!")
    else:
        print(f"Caffe reference model was already downloaded into {model_fname}, skipped")

    prototxt_fname = os.path.join(caffe_model_dir, "deploy.prototxt")

    if not os.path.isfile(prototxt_fname):
        resp = requests.get(r"https://raw.githubusercontent.com/BVLC/caffe/master/models/"
                            r"bvlc_reference_caffenet/deploy.prototxt", stream=True)

        total = int(resp.headers.get('content-length', 0))
        with open(prototxt_fname, 'wb') as file, tqdm(desc="Downloading prototxt for Caffe reference model...",
                                             total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:

            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

        print(f"\nPrototxt for Caffe reference model downloaded into {prototxt_fname}!")
    else:
        print(f"Prototxt for Caffe reference model was already downloaded into {prototxt_fname}, skipped")


def main_comparison():
    dl_tradesy_feedback()
    print("".center(80, '-'))
    dl_visual_feature()


if __name__ == "__main__":
    main()
