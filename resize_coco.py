import os
import argparse
import shutil
import PIL
from PIL import Image


def resize_img(img: PIL.Image.Image, size: list) -> PIL.Image.Image:
    return img.resize(size=size, resample=Image.ANTIALIAS)


def resize_coco(input_path: str, output_path: str, size: list):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    img_list = os.listdir(input_path)
    img_count = len(img_list)
    print('total image :', img_count)
    input('ready to begin')
    for i, img_name in enumerate(img_list):
        with Image.open(os.path.join(input_path, img_name)) as img:
            img = resize_img(img, size)
            img.save(os.path.join(output_path, img_name), img.format)
        if (i + 1) % 500 == 0:
            print('%d/%d images are resized to %s' % (i + 1, img_count, output_path))


def do(args: argparse.Namespace):
    image_size = [args.img_size, args.img_size]
    resize_coco(args.input_path, args.output_path, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tool to resize each image in coco')
    parser.add_argument('--input_path', type=str, default='.\\data\\train2014', help='input path of coco. default : .\\data\\train2014')
    parser.add_argument('--output_path', type=str, default='.\\data\\train2014_resize', help='output path of coco. default : .\\data\\train2014_resize')
    parser.add_argument('--img_size', type=int, default=256, help='size of images to resize. default : 256 x 256')
    do(parser.parse_args())
