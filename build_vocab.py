import nltk
import argparse
import pickle
from collections import Counter
from pycocotools.coco import COCO
from vocabulary import Vocabulary


def build_vocab(json_path: str, threshold: int) -> Vocabulary:
    coco_cls = COCO(json_path)
    countt_cls = Counter()
    ids = coco_cls.anns.keys()
    for i, idt in enumerate(ids):
        caption = str(coco_cls.anns[idt]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        countt_cls.update(tokens)
        if (i + 1) % 1000 == 0:
            print('%d/%d tokenize the captions' % (i + 1, len(ids)))
    words = [word for word, cnt in countt_cls.items() if cnt >= threshold]
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    for word in words:
        vocab.add_word(word)
    return vocab


def do(args: argparse.Namespace):
    vocab = build_vocab(args.caption_path, args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
        f.close()
    print('total vocab size :', len(vocab))
    print('save vocab wrapper to ', vocab_path)


if __name__ == '__main__':
    opt = argparse.ArgumentParser(description='script to build a vocab using the captions from coco dataset')
    opt.add_argument('--caption_path', type=str, default='.\\data\\annotations\\captions_train2014.json', help='coco caption json file path')
    opt.add_argument('--vocab_path', type=str, default='.\\data\\vocab.pickle', help='path of the vocab file to save')
    opt.add_argument('--threshold', type=int, default=4, help='minimum word count threshold')
    args = opt.parse_args()
    print(args)
    do(args)
