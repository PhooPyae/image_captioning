import sys
sys.path.append('/projects/bdfr/plinn/image_captioning/baseline')
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
from PIL import Image
from utils.tokenizer import Vocabulary

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform = None, freq_threshold = 5):
        self.root_dir = root_dir
        # df = pd.read_csv(captions_file)
        df = pd.read_csv(captions_file, sep="\t", header=None, names=["image_caption", "caption"])
    
        # Split the 'image_caption' column into 'image' and 'caption_number'
        df[['image', 'caption_number']] = df['image_caption'].str.split('#', expand=True)
        df = df.drop(columns=['image_caption'])
        self.df = df[['image', 'caption_number', 'caption']]
        
        # Group by image and sample two captions per image
        # df_sampled = df.groupby('image').apply(lambda x: x.sample(min(5, len(x)))).reset_index(drop=True)
        
        # # Split data by unique images (ensuring no overlap in images between train and validation)
        # unique_images = df_sampled['image'].unique()
        # train_images, val_images = train_test_split(unique_images, test_size=0.2, random_state=42)
        
        # # Create training and validation dataframes by selecting based on image
        # train_df = df_sampled[df_sampled['image'].isin(train_images)]
        # val_df = df_sampled[df_sampled['image'].isin(val_images)]
        
        # # Shuffle the training and validation datasets
        # train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        # val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # df['image'] = df['image_name']
        # df['caption'] = df['comment']
        # self.df = df.loc[:,['image', 'caption']]
        
        self.transform = transform
    
        self.images = self.df['image']
        self.captions = self.df['caption']
        
        #init vocab and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
#         print(caption)
        img_id = self.images[idx]
        image = Image.open('/projects/bdfr/plinn/image_captioning/data/Flicker8k_Dataset/'+img_id).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        tokenized_caption = [self.vocab.string_to_index["<SOS>"]]
        tokenized_caption += self.vocab.tokenize(caption)
        tokenized_caption.append(self.vocab.string_to_index["<EOS>"])

        return image, torch.tensor(tokenized_caption)
    
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim = 0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first = False, padding_value = self.pad_idx)

        return images, targets
    