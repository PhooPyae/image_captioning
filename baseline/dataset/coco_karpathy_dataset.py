import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import requests
from PIL import Image
from io import BytesIO
import random
import numpy as np
from utils import pre_caption
import torch

class coco_karpathy_train(Dataset):
    def __init__(self, ann_root, image_root, vocab, max_words=30, prompt='', transform=None):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        filename = 'coco_karpathy_train.json'
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.image_root = image_root

        self.transform = transform
        self.max_words = max_words      
        self.prompt = prompt
        self.vocab = vocab
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image = Image.open(os.path.join(self.image_root), ann['image']).convert('RGB')  
        caption = ann['caption']

        if self.transform:
            image = self.transform(image)
        
        caption = self.prompt+pre_caption(caption, self.max_words) 
        
        tokenized_caption = [self.vocab.string_to_index["<SOS>"]]
        tokenized_caption += self.vocab.tokenize(caption)
        tokenized_caption.append(self.vocab.string_to_index["<EOS>"])

        return image, torch.tensor(tokenized_caption), ann['image']
    
    
class coco_karpathy_caption_eval(Dataset):
    def __init__(self, ann_root, image_root, split, prompt, max_words=50, transform=None):  
        '''
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.image_root = image_root
        self.transform = transform
        self.prompt = prompt
        self.max_words = max_words
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        #choose random caption
        caption = random.choice(ann['caption'])
        
        image = Image.open(os.path.join(self.image_root, ann['image'])).convert('RGB')  
        
        if self.transform:
            image = self.transform(image)
        
        caption = self.prompt+pre_caption(caption, self.max_words) 
        
        return image, caption