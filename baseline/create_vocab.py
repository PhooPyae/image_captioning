import pandas as pd
from tokenizer import Vocabulary
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader, Dataset
import json
import os
import itertools

class DataPreparation:
    def __init__(self, ann_root):
        self.ann_root = ann_root
        self.captions = []
        self.freq_threshold = 5
        
        self.vocab = Vocabulary(self.freq_threshold)

    def create_vocab(self, dataset):
        annotation = json.load(open(os.path.join(self.ann_root,dataset),'r'))
        caption_list = [ann['caption'] for ann in annotation]
        self.captions.extend(caption_list)
    
    def save_tokens(self):
        flatten_captions = list(itertools.chain.from_iterable(self.captions))
        self.vocab.build_vocab(flatten_captions)
        
        with open("index_to_string.json", "w") as outfile: 
            json.dump(self.vocab.index_to_string, outfile)
    
        with open("string_to_index.json", "w") as outfile: 
            json.dump(self.vocab.string_to_index, outfile)
           
    
if __name__ == '__main__':
    ann_root = '/projects/bdfr/plinn/image_captioning/baseline/coco'
    datasets = ['coco_karpathy_train.json', 'coco_karpathy_val.json', 'coco_karpathy_test.json']
    preparation = DataPreparation(ann_root)
    for dataset in datasets:
        preparation.create_vocab(dataset)
        print(f'{dataset}: {len(preparation.captions)}')
    
    preparation.save_tokens()
    print(f'vocab_size {preparation.vocab.__len__()}')
    
    