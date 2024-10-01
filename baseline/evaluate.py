import os
import numpy as np
from PIL import Image
from config import Config
import torch 

config = Config()

def generate(self, model, image, vocabulary, max_length=50):
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

def evaluate_model(model, dataset, transform):
    df = dataset.df
    model.eval()
    img_dir = '/projects/bdfr/plinn/image_captioning/data/Flicker8k_Dataset'
    files = os.listdir(img_dir)
    images = np.random.choice(files, 5)
    for img in images:
        caption = df[df['image'] == img]['caption'].values
        image = Image.open(img_dir+"/"+img).convert("RGB")
        transformed_image = transform(image).unsqueeze(0)
        generated_caption = " ".join(generate(model, transformed_image.to(config.device), dataset.vocab))
        return caption, generated_caption
        
def compute_ble(dataset):
    for image, tokenized_caption in tqdm(dataset):
    # Generate the caption using your model
    generated_caption = " ".join(generate(model, image.unsqueeze(0).to(device), dataset.vocab))
    
    # Decode the actual (reference) caption
    reference_caption = decode(tokenized_caption.tolist(), dataset.vocab)
    
    # Tokenize the generated and reference captions
    reference_tokens = [token for token in reference_caption.split() if token not in ["<SOS>", "<EOS>"]]
    generated_tokens = [token for token in generated_caption.split() if token not in ["<SOS>", "<EOS>"]]
#     print(reference_tokens)
#     print(generated_tokens)
    
    # Calculate the BLEU score
    bleu_score += sentence_bleu([reference_tokens], generated_tokens)
    
    # Print the captions and the BLEU score
    print("Reference Caption:", reference_caption)
    print("Generated Caption:", generated_caption)
    print("BLEU Score:", bleu_score)
    avg_bleu_score = bleu_score / dataset.__len__()
    print(f'Average BELU {avg_bleu_score}')
    
    # Optional: break after the first example
