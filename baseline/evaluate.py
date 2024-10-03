import os
import numpy as np
from PIL import Image
from model import CNNtoRNN
from config import Config
import torch 
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from dataset import FlickrDataset
import torchvision.transforms as transforms
from utils import *
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm
import wandb
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()

def generate(model, image, vocabulary, max_length=50):
    result_caption = []

    with torch.no_grad():
        x = model.encoderCNN(image).unsqueeze(0)
        states = None

        for _ in range(max_length):
            hiddens, states = model.decoderRNN.lstm(x, states)
            output = model.decoderRNN.linear(hiddens.squeeze(0))
            predicted = output.argmax(1)
#                 print(predicted.shape)
            result_caption.append(predicted.item())
            x = model.decoderRNN.embed(predicted).unsqueeze(0)

            if vocabulary.index_to_string[predicted.item()] == "<EOS>":
                break

    return [vocabulary.index_to_string[idx] for idx in result_caption]

def generate_coco(model, image, vocabulary, max_length=50):
    result_caption = []

    with torch.no_grad():
        x = model.encoderCNN(image).unsqueeze(0)
        states = None

        for _ in range(max_length):
            hiddens, states = model.decoderRNN.lstm(x, states)
            output = model.decoderRNN.linear(hiddens.squeeze(0))
            predicted = output.argmax(1)
#                 print(predicted.shape)
            result_caption.append(predicted.item())
            x = model.decoderRNN.embed(predicted).unsqueeze(0)

            if vocabulary.index_to_string[str(predicted.item())] == "<EOS>":
                break

    return [vocabulary.index_to_string[str(idx)] for idx in result_caption]

def evaluate_model(model, dataset, vocab, transform, data='flickr8k'):
    model.eval()
    if data == 'flickr8k':
        df = dataset.df
        img_dir = '/projects/bdfr/plinn/image_captioning/data/Flicker8k_Dataset'
        images = os.listdir(img_dir)
        images = np.random.choice(files, 5)
        for img in images:
            caption = df[df['image'] == img]['caption'].values
            image = Image.open(img_dir+"/"+img).convert("RGB")
            transformed_image = transform(image).unsqueeze(0)
            generated_caption = " ".join(generate(model, transformed_image.to(config.device), vocab))
            logger.info(f'Ground Truth Caption: {caption}')
            logger.info(f'Generated Caption: {generated_caption}')

    elif data == 'coco':
        indices = np.random.randint(0, dataset.__len__(), 5)
        for i in indices:
            transformed_image, caption = dataset.__getitem__(i)
            generated_caption = " ".join(generate_coco(model, transformed_image.unsqueeze(0).to(config.device), vocab))
            logger.info(f'Ground Truth Caption: {caption}')
            logger.info(f'Generated Caption: {generated_caption}') 
       
    return caption, generated_caption

def decode(token_indices, vocab):
        """
        Convert a list of token indices into a string of words.
        
        Args:
            token_indices (list): List of token indices to convert.
        
        Returns:
            str: The decoded sentence.
        """
        words = []
        for index in token_indices:
            word = vocab.index_to_string.get(index, "<UNK>")
#             if word == "<EOS>":
#                 break
            words.append(word)
        
        return ' '.join(words)
    
# def compute_bleu(model, dataset, is_last=False):
#     bleu_score = 0
#     if is_last:
#         for image, tokenized_caption in dataset:
#             generated_caption = " ".join(generate(model, image.unsqueeze(0).to(config.device), dataset.vocab))
        
#             # Decode the actual (reference) caption
#             reference_caption = decode(tokenized_caption.tolist(), dataset.vocab)
            
#             # Tokenize the generated and reference captions
#             reference_tokens = [token for token in reference_caption.split() if token not in ["<SOS>", "<EOS>"]]
#             generated_tokens = [token for token in generated_caption.split() if token not in ["<SOS>", "<EOS>"]]
#         #     print(reference_tokens)
#         #     print(generated_tokens)
        
#             # Calculate the BLEU score
#             bleu_score += sentence_bleu([reference_tokens], generated_tokens)
#             logger.info(f"Reference Caption: {reference_caption}")
#             logger.info(f"Generated Caption: {generated_caption}")
#             logger.info(f"BLEU Score: {bleu_score}")
            
#         return bleu_score / dataset.__len__() 
        
#     else:
#         indices = np.random.randint(len(dataset), size=5)
#         for i in indices:
#             image, tokenized_caption = dataset.__getitem__(i)
#             # Generate the caption using your model
#             generated_caption = " ".join(generate(model, image.unsqueeze(0).to(config.device), dataset.vocab))
            
#             # Decode the actual (reference) caption
#             reference_caption = decode(tokenized_caption.tolist(), dataset.vocab)
            
#             # Tokenize the generated and reference captions
#             reference_tokens = [token for token in reference_caption.split() if token not in ["<SOS>", "<EOS>"]]
#             generated_tokens = [token for token in generated_caption.split() if token not in ["<SOS>", "<EOS>"]]
#         #     print(reference_tokens)
#         #     print(generated_tokens)
            
#             # Calculate the BLEU score
#             bleu_score += sentence_bleu([reference_tokens], generated_tokens)
            
#             # Print the captions and the BLEU score
#             logger.info(f"Reference Caption: {reference_caption}")
#             logger.info(f"Generated Caption: {generated_caption}")
#             logger.info(f"BLEU Score: {bleu_score}")

#         avg_bleu_score = bleu_score / dataset.__len__()
#         return avg_bleu_score
    
    # Optional: break after the first example

if __name__ == '__main__':
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    model_dir = '/projects/bdfr/plinn/image_captioning/'
    dataset = FlickrDataset('/projects/bdfr/plinn/image_captioning/data/Flicker8k_Datasets', captions_file = '/projects/bdfr/plinn/image_captioning/data/Flickr8k.token.txt', transform = transform)
    vocab_size = len(dataset.vocab)
    base_model = CNNtoRNN(config.embed_size, config.hidden_size, vocab_size, config.num_layers).to(config.device)
    model_files = ['my_checkpoint.pth_epoch50.tar', 'my_checkpoint.pth_epoch100.tar', 'my_checkpoint.pth_epoch150.tar']
    # avg_bleu_score = compute_bleu(model, dataset, is_last=True)
    # print(f'Avreage bleu score {avg_bleu_score}')
    # evaluate_model(model, dataset, transform)
    bleu_score = []
    df = dataset.df
    img_dir = '/projects/bdfr/plinn/image_captioning/data/Flicker8k_Dataset'
    images = os.listdir(img_dir)
    wandb.init(project="evaluation_image_captioning")
    def compute_bleu(model):
        model.eval()
        for i, img in tqdm(enumerate(images)):
            captions = df[df['image'] == img]['caption'].values
            image = Image.open(img_dir+"/"+img).convert("RGB")
            transformed_image = transform(image).unsqueeze(0)
            generated_caption = " ".join(generate(model, transformed_image.to(config.device), dataset.vocab))

            reference_tokens = []
            for caption in captions:
                reference_tokens.append([token.lower() for token in caption.split() if token not in ["<SOS>", "<EOS>"]])
                
            generated_tokens = [token.lower() for token in generated_caption.split() if token not in ["<SOS>", "<EOS>"]]

            bscore = sentence_bleu(reference_tokens, generated_tokens, weights=(0.4, 0.4, 0.15, 0.05))
            wandb.log({
                "bleu score": bscore, 
                "image_index": i
                })
            if bscore < 0.5:
                logger.info(f"Reference Caption: {captions}")
                logger.info(f"Generated Caption: {generated_caption}")
                logger.info(f"bleu score: {bscore:.4f}")
            bleu_score.append(bscore)
        avg_bleu_score = np.mean(bleu_score)
        logger.info(f'Average BLEU score {avg_bleu_score:.4f}')
        return avg_bleu_score

        
    for model_file in model_files:
        model_ = load_state(torch.load(model_dir + model_file), base_model)
        logger.info(f'Evaluating Model {model_file}')
        average_bleu_score = compute_bleu(model_)
        wandb.log({
            "average_bleu_score": average_bleu_score, 
            "model": model_file
            })