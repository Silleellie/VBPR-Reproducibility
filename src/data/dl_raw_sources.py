import gzip
import os
import shutil

import gdown
import requests
from tqdm import tqdm

from src import RAW_DIR


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


def main():
    dl_tradesy_feedback()
    print("".center(80, '-'))
    dl_visual_feature()


if __name__ == "__main__":
    main()
