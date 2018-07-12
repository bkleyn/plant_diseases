"""Pre-process images for training SRCNN model."""

from os import listdir, makedirs
from os.path import isfile, join, exists

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="Data input directory")
parser.add_argument("output_dir", help="Data output directory")
args = parser.parse_args()

import numpy as np
from scipy import misc

SCALE = 8.0
INPUT_SIZE = 33
LABEL_SIZE = 21
STRIDE = 14
PAD = int((INPUT_SIZE - LABEL_SIZE) / 2)

if not exists(args.output_dir):
    makedirs(args.output_dir)
if not exists(join(args.output_dir, "input")):
    makedirs(join(args.output_dir, "input"))
if not exists(join(args.output_dir, "label")):
    makedirs(join(args.output_dir, "label"))

count = 1
for f in listdir(args.input_dir):
    f = join(args.input_dir, f)
    if not isfile(f):
        continue

    im = misc.imread(f, flatten=False, mode='RGB')

    w, h, c = im.shape
    w = int(w - w % SCALE)
    h = int(h - h % SCALE)
    im = im[0:w, 0:h]

    scaled = misc.imresize(im, 1.0/SCALE, 'bicubic')
    scaled = misc.imresize(scaled, SCALE/1.0, 'bicubic')

    for i in range(0, h - INPUT_SIZE + 1, STRIDE):
        for j in range(0, w - INPUT_SIZE + 1, STRIDE):
            sub_img = scaled[j : j + INPUT_SIZE, i : i + INPUT_SIZE, :]
            sub_img_label = im[j + PAD : j + PAD + LABEL_SIZE, i + PAD : i + PAD + LABEL_SIZE,:]
            misc.imsave(join(args.output_dir, "input", str(count) + '.jpg'), sub_img)
            misc.imsave(join(args.output_dir, "label", str(count) + '.jpg'), sub_img_label)

            count += 1
