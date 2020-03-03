from PIL import Image
import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from vocabulary import Vocabulary
from nltk.tokenize import word_tokenize


class CocoDataset(data.Dataset):
    def __init__(self, root: str, json_path: str, vocab: Vocabulary, transform: transforms = None):
        self.root = root
        self.coco_cls = COCO(json_path)
        self.coco_idx = list(self.coco_cls.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        coco_id = self.coco_idx[index]
        img_id = self.coco_cls.anns[coco_id]['image_id']
        caption = self.coco_cls.anns[coco_id]['caption']
        img_name = self.coco_cls.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        words = word_tokenize(str(caption).lower())
        tokens = []
        tokens.append(self.vocab('<start>'))
        tokens.extend([self.vocab(word) for word in words])
        tokens.append(self.vocab('<end>'))
        return img, torch.Tensor(tokens)

    def __len__(self):
        return len(self.coco_idx)


def get_data(data: list) -> tuple:
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, dim=0)
    lengths = [len(x) for x in captions]
    captions_arr = torch.zeros(size=(len(captions), max(lengths)), dtype=torch.long)
    for i, cap in enumerate(captions):
        captions_arr[i, :lengths[i]] = cap[:lengths[i]]
    return images, captions_arr, lengths


def get_dataloader(root: str, json_path: str, vocab: Vocabulary, batch_size: int, num_workers: int, transform: transforms = None, shuffle: bool = False):
    coco_set = CocoDataset(root, json_path, vocab, transform)
    coco_loader = data.DataLoader(dataset=coco_set, batch_size=batch_size, shuffle=shuffle, collate_fn=get_data, num_workers=num_workers)
    return coco_loader


if __name__ == '__main__':
    print('just a simple')
