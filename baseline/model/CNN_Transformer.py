import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
import sys

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
#         self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Linear layer to reduce feature dimensions
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
#         self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
#         print('>>>>> ENCODER RNN <<<<<<')
        features = self.resnet(images)
#         print(f'output of resnet {features.shape}')
        features = features.view(features.size(0), -1)
#         print(f'after reshape {features.shape}')
        features = self.embed(features)
#         print(f'after self.embed {features.shape}')
        return self.dropout(self.relu(features))

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_size))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(1)

    def forward(self, x):
        # x shape: (seq_len, batch_size, embed_size)
        seq_len = x.size(0)
        return x + self.encoding[:seq_len, :].to(x.device)
    
class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, num_heads, dropout=0.5, max_len=100):
        super(DecoderTransformer, self).__init__()

        # Embedding layers for tokens and positional encoding
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Embedding(max_len, embed_size)

        # Transformer decoder layers
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        
        # Output layer to project the output back to vocabulary size
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, captions, features, tgt_mask=None):
        # Get the shape (batch_size, seq_len)
        batch_size, seq_len = captions.shape

        # Embedding for the captions
        embeddings = self.dropout(self.embed(captions))  # (batch_size, seq_len, embed_size)
        print(f'{embeddings.shape=}')

        # Positional encodings (need to expand across batch)
        pos_encodings = self.positional_encoding(torch.arange(seq_len, device=features.device)).unsqueeze(0)  # (1, seq_len, embed_size)
        print(f'{pos_encodings.shape=}')
        embeddings += pos_encodings  # Add positional encoding to embeddings

        print(f'after concat: {embeddings.shape}')
        sys.exit(1)
        # Expand image features to fit across sequence
        features = features.unsqueeze(1).repeat(1, seq_len, 1)  # Repeat the image features across the sequence

        # Transformer decoder forward pass
        output = self.transformer_decoder(embeddings.transpose(0, 1), features.transpose(0, 1), tgt_mask=tgt_mask)
        output = self.fc_out(output.transpose(0, 1))  # Project to vocabulary size

        return output
        
        
class CNNtoTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, num_heads, max_sequence_length, device):
        super(CNNtoTransformer, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size).to(device)
        self.decoderTransformer = DecoderTransformer(
            embed_size=embed_size, 
            hidden_size=hidden_size, 
            vocab_size=vocab_size, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            dropout=0.5, 
            max_len=max_sequence_length
        ).to(device)

    def forward(self, images, captions):
#         print('>>>>> CNN to Transformer <<<<<<')
        features = self.encoderCNN(images)
        seq_len, batch_size = captions.shape  # seq_len = 18, batch_size = 32
        # Expand the features to have the same sequence length as the captions
        features = features.unsqueeze(1).repeat(1, seq_len, 1)  # Now features shape will be (32, 18, 256)
        # Transpose captions to (batch_size, seq_len) -> (seq_len, batch_size)
        captions = captions.transpose(0, 1)
        print(f'{features.shape=}')
        print(f'{captions.shape=}')
        outputs = self.decoderTransformer(captions, features)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            x = x.unsqueeze(0)

            input_ids = torch.tensor([[vocabulary.string_to_index["<SOS>"]]]).to(image.device)

            for _ in range(max_length):
                outputs = self.decoderTransformer(x, input_ids)
                predicted = outputs[:,-1,:].argmax(1)
#                 print(predicted.shape)
                result_caption.append(predicted.item())
                input_ids = torch.cat((input_ids, predicted.unsqueeze(0).unsqueeze(-1)), dim=1)

                if vocabulary.index_to_string[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.index_to_string[idx] for idx in result_caption]