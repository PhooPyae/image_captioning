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

class Transformer(nn.Module):
    

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
#         print('>>>>> CNN to RNN <<<<<<')
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
#                 print(predicted.shape)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.index_to_string[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.index_to_string[idx] for idx in result_caption]