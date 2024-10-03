import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dataset import MyCollate
from PIL import Image
import re

def load_data(data_path):
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

def get_loader(
    dataset,
    batch_size = 32,
    num_workers = 4,
    shuffle = True,
    pin_memory = True
):
    pad_idx = dataset.vocab.string_to_index["<PAD>"]
    
    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory,
        collate_fn = MyCollate(pad_idx = pad_idx))
    
    return loader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_state(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return model

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

def print_examples(model, device, dataset, path, transform):
    model.eval()
    for i in range(5):
        idx = np.random.choice(dataset.df.index)
        img = dataset.df.loc[idx, 'image']
        caption = dataset.df.loc[idx, 'caption']
        image = Image.open(path+"/"+img).convert("RGB")
        transformed_image = transform(image).unsqueeze(0)
        generated_caption = " ".join(model.caption_image(transformed_image.to(device), dataset.vocab))
        print(f'Ground Truth Caption: {caption}')
        print(f'Generated Caption: {generated_caption}')

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption