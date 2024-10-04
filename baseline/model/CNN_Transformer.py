import torch
import torchvision.models as models
import torch.nn as nn

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
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
    
class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, num_heads, max_length, device):
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=0.5)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.device = device
        
    def forward(self, features, captions):
        B, T = captions.shape
        pos_mask = torch.triu(torch.ones((T, T)), diagonal=1).bool().to(self.device)
        embeddings = self.dropout(self.embed(captions))
        embeddings = self.pos_encoder(embeddings)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        
        outputs = self.transformer(embeddings, src_mask=pos_mask)
        outputs = self.linear(outputs)
        return outputs
        
        
class CNNtoTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, num_heads, max_length, device):
        super(CNNtoTransformer, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size).to(device)
        self.decoderTransformer = DecoderTransformer(embed_size, hidden_size, vocab_size, num_layers, num_heads, max_length, device)

    def forward(self, images, captions):
#         print('>>>>> CNN to Transformer <<<<<<')
        features = self.encoderCNN(images)
        outputs = self.decoderTransformer(features, captions)
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