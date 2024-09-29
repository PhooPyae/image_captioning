import os
import pandas as pd
import spacy
from PIL import Image
import wandb
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from config import Config

from model import CNNtoRNN

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config = Config()
spacy_eng = spacy.load("en_core_web_sm")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
#     transforms.RandomResizedCrop((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def train():
    step = 0
    # Only finetune the CNN
    for name, param in model.encoderCNN.resnet.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    best_loss = float('inf')
    for epoch in range(num_epochs):

        total_loss = 0

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(
            enumerate(dataloader), total=len(dataloader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
    #         print(outputs.shape)
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
        
            total_loss += loss
            
            if idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
                wandb.log({"train_loss": loss.item(), "step": step, "epoch": epoch+1, "learning_rate": optimizer.param_groups[0]["lr"]})

            step += 1
            
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
            
            average_loss = total_loss / len(dataloader)
    #         print(f'Epoch [{epoch+1}/{num_epochs}]], Loss: {average_loss:.4f}')
            
            if average_loss < best_loss:
                best_loss = average_loss
                torch.save(model.state_dict(), f'best_model.pth')
    #         else:
    #             print(f'Early stopping | Loss : {best_loss}')

if __name__ == '__main__':
    dataset = FlickrDataset('/projects/bdfr/plinn/image_captioning/data/Flicker8k_Datasets', captions_file = '/kaggle/input/flickr30k/captions.txt', transform = transform)
    dataloader = get_loader(dataset)

    logger.debug('Loaded Dataset !')
    logger.debug(f'Train data: {dataset.__len__()}')
    logger.debug(dataset.__getitem__(0))

    wandb.init(
        project="image_captioning_CNN_RNN",
        config={
        "learning_rate": config.learning_rate,
        "epochs": config.num_epochs,
        "batch_size": 32,
        "optimizer": "Adam",
        "image_encoder": "ResNet50",
        "text_decoder": "RNN",
        "n_layers": config.num_layers,
        "embed_size": config.embed_size,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size
        }
    )

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.string_to_index["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Total number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    train()