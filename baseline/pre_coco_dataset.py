import os
import requests
from pycocotools.coco import COCO
from tqdm import tqdm
import re
import json

def download_annotation(annotations_zip_url, annotations_path, annotations_file):
    os.makedirs(annotations_path, exist_ok=True)

    # Download and extract annotations if not already done
    if not os.path.exists(annotations_file):
        print("Downloading annotations...")
        with requests.get(annotations_zip_url, stream=True) as r:
            r.raise_for_status()
            with open(annotations_file, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), desc="Downloading annotations"):
                    f.write(chunk)
        os.system(f'unzip -q {annotations_file} -d {annotations_path}')

def download_images(image_id, path):
    os.makedirs(path, exist_ok=True)
    image_url = 'http://images.cocodataset.org/' + image_id
    img_filename = os.path.join(path, f"{image_id}")
    if not os.path.exists(img_filename):
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(img_filename, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
                    
if __name__ == '__main__':
    urls = {
        'train': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json',
        'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
        'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
    
    data_path = '/projects/bdfr/plinn/image_captioning/baseline/coco'
    download_annotation(urls['train'], data_path, 'coco_karpathy_train.json')
    download_annotation(urls['val'], data_path, 'coco_karpathy_val.json')
    download_annotation(urls['test'], data_path, 'coco_karpathy_test.json')
    
    annotation = json.load(open(os.path.join(data_path,'coco_karpathy_train.json'),'r'))
    for ann in tqdm(annotation):
        download_images(ann['image'], data_path)