import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.encoder = nn.Sequential(*(list(resnet.children())[:-1]))
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=embed_size)
        self.bn1d = nn.BatchNorm1d(num_features=embed_size, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        out = out.reshape(out.size(0), -1)
        return self.bn1d(self.fc(out))


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, layer_num: int, max_sample_length=31):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=layer_num, bias=True, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.max_length = max_sample_length

    def forward(self, img_features: torch.Tensor, caption: torch.Tensor, length: list) -> torch.Tensor:
        embed = self.embed(caption)
        concated = torch.cat([embed, img_features.unsqueeze(1)], dim=1)  # why must unsqueeze(1) here?
        lstm_pack = pack_padded_sequence(concated, length, batch_first=True)
        out, _ = self.lstm(lstm_pack)
        return self.fc(out[0])

    def sample(self, img_features: torch.Tensor, state: tuple = None) -> torch.Tensor:
        sample_ids = []
        inputs = img_features.unsqueeze(1)
        for i in range(self.max_length):
            hidden, state = self.lstm(inputs, state)
            out = self.fc(hidden.squeeze(1))
            _, predicted = torch.max(out, dim=1)
            sample_ids.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)
        sample_ids = torch.stack(sample_ids, dim=1)
        return sample_ids[0]


if __name__ == '__main__':
    encoder = EncoderCNN(100)
    decoder = DecoderRNN(2000, 200, 1024, 3)
    print('It\'s just a sample!')
    print(encoder)
    print(decoder)
    print('params:')
    print(list(encoder.parameters()))
    print(list(decoder.parameters()))
