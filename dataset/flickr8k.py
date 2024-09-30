import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path):
    # df=  pd.read_csv(data_path)
    df = pd.read_csv(data_path, sep="\t", header=None, names=["image_caption", "caption"])
    df[['image', 'caption_number']] = df['image_caption'].str.split('#', expand=True)
    df = df.drop(columns=['image_caption'])
    df = df[['image', 'caption_number', 'caption']]
    logger.info(df.head())
    train_df , val_df = train_test_split(df , test_size = 0.2)
    train_df.to_csv('train_df.csv')
    val_df.to_csv('val_df.csv')
    return train_df, val_df

def load_data_shuffle(data_path):
    # Read the dataset
    df = pd.read_csv(data_path, sep="\t", header=None, names=["image_caption", "caption"])
    
    # Split the 'image_caption' column into 'image' and 'caption_number'
    df[['image', 'caption_number']] = df['image_caption'].str.split('#', expand=True)
    df = df.drop(columns=['image_caption'])
    df = df[['image', 'caption_number', 'caption']]
    
    # Group by image and sample two captions per image
    df_sampled = df.groupby('image').apply(lambda x: x.sample(min(5, len(x)))).reset_index(drop=True)
    
    # Split data by unique images (ensuring no overlap in images between train and validation)
    unique_images = df_sampled['image'].unique()
    train_images, val_images = train_test_split(unique_images, test_size=0.2, random_state=42)
    
    # Create training and validation dataframes by selecting based on image
    train_df = df_sampled[df_sampled['image'].isin(train_images)]
    val_df = df_sampled[df_sampled['image'].isin(val_images)]
    
    # Shuffle the training and validation datasets
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Logging for inspection
    logger.info(f"Train data: {train_df.shape}, Validation data: {val_df.shape}")
    logger.info(f"Sample of training data: \n{train_df.head()}")
    
    # Save train and validation sets to CSV
    train_df.to_csv('train_df.csv', index=False)
    val_df.to_csv('val_df.csv', index=False)
    
    return train_df, val_df
    
class ImgDataset(Dataset):
    def __init__(self, df,root_dir,tokenizer,feature_extractor, transform = None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer= tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = 50
        
    def __len__(self,):
        return len(self.df['image'].unique())
    
    def __getitem__(self,idx):
        image_id = self.df['image'].unique()[idx]
        captions = self.df[self.df['image'] == image_id]['caption'].tolist()

        caption = random.choice(captions)
        
        img_path = os.path.join(self.root_dir , image_id)
        img = Image.open(img_path).convert("RGB")
    
        if self.transform:
            img= self.transform(img)

        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        caption = caption.lower()
        captions = self.tokenizer(caption,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  truncation=True,
                                  return_tensors='pt').input_ids.squeeze()
        labels = torch.where(tokenized_caption == self.tokenizer.pad_token_id, torch.tensor(-100), tokenized_caption)
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": captions}
        return encoding
    
if __name__ == '__main__':
    train_df, val_df = load_data_shuffle(data_path = '/projects/bdfr/plinn/image_captioning/data/Flickr8k.token.txt')