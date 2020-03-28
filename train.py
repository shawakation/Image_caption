import argparse
import pickle

import numpy as np
import torch.nn as nn
# import torch.utils.tensorboard as tensorboard
from torch.nn.utils.rnn import pack_padded_sequence

from data_loader import *
from models import EncoderCNN, DecoderRNN

device = torch.device('cuda:0')


def do(args: argparse.Namespace):
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    # writer = tensorboard.SummaryWriter('.\\records')
    # preprocess
    preprocess = transforms.Compose([
        transforms.RandomCrop(args.random_crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # dataset
    coco_loader = get_dataloader(root=args.dataset_path, json_path=args.json_path, vocab=vocab, batch_size=args.batch_size, num_workers=args.num_workers,
                                 transform=preprocess, shuffle=False)
    # models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(len(vocab), args.embed_size, args.hidden_size, args.num_layers).to(device)
    loss_cls = nn.CrossEntropyLoss().to(device)
    params = list(encoder.fc.parameters()) + list(encoder.bn1d.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    # resume
    if args.resume:
        model_states = torch.load(os.path.join(args.save_model_path, 'model.ckpt'))
        print('checkpoint epoch: %d\tstep: %d' % (model_states['epoch'], model_states['step']))
        encoder.load_state_dict(model_states['encoder'])
        decoder.load_state_dict(model_states['decoder'])
        print('load successfully')
    # train
    total_step = len(coco_loader)
    print('total step in each epoch : ', total_step)
    encoder.fc.train(mode=True)
    encoder.bn1d.train(mode=True)
    encoder.encoder.eval()
    decoder.train(mode=True)
    input('ready')
    for cur_epoch in range(args.num_epochs):
        for cur_step, (image, caption, length) in enumerate(coco_loader):
            image = image.to(device)
            caption = caption.to(device)
            target = pack_padded_sequence(caption, length, batch_first=True)[0]
            out = decoder(encoder(image), caption, length)
            loss = loss_cls(out, target)
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()
            # input('pause test')
            if (cur_step + 1) % args.print_step == 0:
                print('Epoch : %d/%d\tStep : %d/%d\tLoss : %.8f\tPerplexity : %.8f' % (
                    cur_epoch + 1, args.num_epochs, cur_step + 1, total_step, loss.item(), np.exp(loss.item())))
            if (cur_step + 1) % args.save_model_step == 0:
                torch.save({'epoch': cur_epoch + 1, 'step': cur_step + 1, 'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
                           os.path.join(args.save_model_path, 'model.ckpt'))
                print('model saved at E:%d\tS:%d' % (cur_epoch + 1, cur_step + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # load and train params
    parser.add_argument('--random_crop_size', type=int, default=224, help='size of randomly cropping images')
    parser.add_argument('--save_model_path', type=str, default='.\\checkpoints', help='path to save models when training')
    parser.add_argument('--dataset_path', type=str, default='.\\data\\train2014_resize', help='path for coco train dataset resized')
    parser.add_argument('--json_path', type=str, default='.\\data\\annotations\\captions_train2014.json', help='path for caption\'s json path')
    parser.add_argument('--vocab_path', type=str, default='.\\data\\vocab.pickle', help='path for vocab.pickle')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--print_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_model_step', type=int, default=1000, help='step size for saving trained models')
    # model params
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=5, help='number of layers in lstm')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=3, help='number of workers of dataloader')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--resume', type=bool, default=False, help='resume model?')
    args = parser.parse_args()
    do(args)
