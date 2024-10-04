import requests
from tqdm import tqdm
from PIL import Image
import os
import json

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
    
    filename = 'coco_karpathy_train.json'
    annotation = json.load(open("/projects/bdfr/plinn/image_captioning/baseline/coco/coco_karpathy_train.json",'r'))
        
    image_urls = list(set([ann['image'] for ann in annotation]))
    
    for image_id in image_urls:
        download_images(image_id, '/projects/bdfr/plinn/image_captioning/images/')
        