import argparse
import math
import sys

import chainer
import numpy
import six
from niconico_chainer_models.google_net import GoogLeNet
from PIL import Image, ImageFile


def resize(img, size):
    h, w = img.size
    ratio = size / float(min(h, w))
    h_ = int(math.ceil(h * ratio))
    w_ = int(math.ceil(w * ratio))
    img = img.resize((h_, w_))
    return img


def fetch_image(url):

    response = six.moves.urllib.request.urlopen(url)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(response)

    if img.mode != 'RGB':  # not RGB
        img = img.convert('RGB')

    img = resize(img, 224)

    x = numpy.asarray(img).astype('f')
    x = x[:224, :224, :3]  # crop

    x /= 255.0  # normalize
    x = x.transpose((2, 0, 1))
    return x


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--model', default='model.npz')
parser.add_argument('--tags', default='tags.txt')
parser.add_argument('image_url')


if __name__ == '__main__':

    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        xp = chainer.cuda.cupy
    else:
        xp = numpy

    # load model
    sys.stderr.write("\r model loading...")
    model = GoogLeNet()
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        model.to_gpu()

    # load tags
    tags = [line.rstrip() for line in open(args.tags)]
    tag_dict = dict((i, tag) for i, tag in enumerate(tags))

    # load image
    sys.stderr.write("\r image fetching...")
    x = xp.array([fetch_image(args.image_url)])
    z = xp.zeros((1, 8)).astype('f')

    sys.stderr.write("\r tag predicting...")
    predicted = model.tag(x, z).data[0]

    sys.stderr.write("\r")
    top_10 = sorted(enumerate(predicted), key=lambda index_value: -index_value[1])[:10]

    for tag, score in top_10:
        if tag in tag_dict:
            tag_name = tag_dict[tag]
            print("tag: {} / score: {}".format(tag_name, score))
