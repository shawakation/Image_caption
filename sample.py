import argparse
import pickle

import torch
import torchvision.transforms as transforms
from PIL import Image

from models import EncoderCNN, DecoderRNN

device = torch.device('cuda:0')


def load_image(img_path: str, transform: transforms = None) -> torch.Tensor:
    image = Image.open(img_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def do(args: argparse.Namespace):
    # preprocess
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # vocab
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # model
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(len(vocab), args.embed_size, args.hidden_size, args.num_layers).to(device)
    model_state = torch.load(args.checkpoint_path, map_location={'cuda:2': 'cuda:0'})
    encoder.load_state_dict(model_state['encoder'])
    decoder.load_state_dict(model_state['decoder'])
    print('load successfully at\tepoch:%d\tstep:%d' % (model_state['epoch'], model_state['step']))
    encoder.eval()
    decoder.eval()
    # image
    img = load_image(args.img_path, preprocess).to(device)
    outs = decoder.sample(encoder(img))
    outs = outs.cpu().numpy()
    print(outs)
    # caption
    caption = []
    for word_id in outs:
        word = vocab.idx2word[word_id]
        caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(caption)
    print(sentence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help='input image path')
    parser.add_argument('--checkpoint_path', type=str, default='.\\checkpoints\\model.ckpt', help='path for trained model')
    parser.add_argument('--vocab_path', type=str, default='.\\data\\vocab.pickle', help='path for vocabulary wrapper')
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
    args = parser.parse_args()
    do(args)
