import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):


    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # disable learning for parameters
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):

    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))


    def forward(self, features, captions):
        cap_embedding = self.embed(
            captions[:, :-1]
        )
        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)
        lstm_out, self.hidden = self.lstm(
            embeddings
        )
        outputs = self.linear(lstm_out)

        return outputs


    def sample(self, inputs, states=None, max_len=20):
        res = []

        for i in range(max_len):
            lstm_out, states = self.lstm(
                inputs, states
            )
            outputs = self.linear(lstm_out.squeeze(dim=1))
            _, predicted_idx = outputs.max(dim=1)
            res.append(predicted_idx.item())
            if predicted_idx == 1:
                break
            inputs = self.embed(predicted_idx)
            inputs = inputs.unsqueeze(1)

        return res
