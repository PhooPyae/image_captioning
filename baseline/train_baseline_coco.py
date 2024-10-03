import os
import pandas as pd
from PIL import Image
import wandb
import numpy as np
import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from config import Config, CocoConfig

from tokenizer import Vocabulary
from model import CNNtoRNN
from coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval
from utils import *
import sys
from evaluate import *
from create_vocab import DataPreparation

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

config = Config()
coco_config = CocoConfig()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

if __name__ == '__main__':
    # Prepare tokenizer for all captions [train, val, test]
    # uncomment only for the first run
    # prepare_data = DataPreparation(coco_config.ann_root)
    # datasets = (coco_config.train, coco_config.val, coco_config.test)
    # for dataset in datasets:
    #     preparation.create_vocab(dataset)
    #     print(f'{dataset}: {len(preparation.captions)}')
    # preparation.save_tokens()
    
    vocab = Vocabulary(freq_threshold=5)
    vocab.index_to_string = json.load(open('/projects/bdfr/plinn/image_captioning/index_to_string.json','r'))
    vocab.string_to_index = json.load(open('/projects/bdfr/plinn/image_captioning/string_to_index.json','r'))
    
    coco_train_dataset = coco_karpathy_train(transform, coco_config.ann_root, vocab, max_words=50, prompt='this image shows ')
    coco_val_dataset = coco_karpathy_caption_eval(transform, coco_config.ann_root, split='val', prompt='this image shows ')
    
    vocab_size = vocab.__len__()
    print(f'{vocab_size=}')
    dataloader = get_loader(coco_train_dataset)

    logger.info('Loaded Dataset !')
    logger.info(f'Train data: {coco_train_dataset.__len__()}')
    logger.info(coco_train_dataset.__getitem__(0))
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
        "vocab_size": vocab_size
        }
    )

    model = CNNtoRNN(config.embed_size, config.hidden_size, vocab_size, config.num_layers).to(config.device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.string_to_index["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Total number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    # model = load_state(torch.load('/projects/bdfr/plinn/image_captioning/my_checkpoint.pth.tar'), model)
    
    step = 0
    # Only finetune the CNN
    for name, param in model.encoderCNN.resnet.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = config.train_CNN

    if config.load_model:
        step = load_checkpoint(torch.load("/projects/bdfr/plinn/image_captioning/my_checkpoint.pth_epoch100.tar"), model, optimizer)

    model.train()

    best_loss = float('inf')
    for epoch in range(config.num_epochs):

        total_loss = 0

        if config.save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(
            enumerate(dataloader), total=len(dataloader), leave=False
        ):
            imgs = imgs.to(config.device)
            captions = captions.to(config.device)
            print(imgs.shape)
            print(captions.shape)

            outputs = model(imgs, captions[:-1])
    #         print(outputs.shape)
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
        
            total_loss += loss
            
            if idx % 100 == 0:
                logger.info(f'Epoch [{epoch+1}/{config.num_epochs}], Step [{idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
            
            if (idx+1) % 1000 == 0:
                caption, generated_caption = evaluate_model(model, coco_val_dataset, vocab, transform, data='coco')
                # avg_bleu_score = compute_bleu(model, dataset)
                # logger.info(f'Average BELU {avg_bleu_score}')
                model.train()
                wandb.log(
                    {
                        "train_loss": loss.item(), 
                        "step": step, 
                        "epoch": epoch+1, 
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        # "avg_bleu_score": avg_bleu_score
                    }
                )

            step += 1
            
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
            
            average_loss = total_loss / len(dataloader)
    #         print(f'Epoch [{epoch+1}/{num_epochs}]], Loss: {average_loss:.4f}')
            
            if average_loss < best_loss:
                best_loss = average_loss
                torch.save(model.state_dict(), f'best_model.pth')
    
    # average_bleu = compute_bleu(model, dataset, is_last=True)
    # logger.info(f'Average BLEU : {average_bleu}')