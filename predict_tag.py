import niconico_chainer_models
import pickle
import argparse
import urllib2
import numpy
import PIL.Image
import chainer
def fetch_image(url):
    response = urllib2.urlopen(url)
    image = numpy.asarray(PIL.Image.open(response).resize((224,224)), dtype=numpy.float32)
    if (not len(image.shape)==3): # not RGB
        image = numpy.dstack((image, image, image))
    if (image.shape[2]==4): # RGBA
        image = image[:,:,:3]
    return image

def to_bgr(image):
    return image[:,:,[2,1,0]]
    return numpy.roll(image, 1, axis=-1)

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("mean")
parser.add_argument("tags")
parser.add_argument("image_url")
parser.add_argument("--gpu", type=int, default=-1)
args = parser.parse_args()

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy

model = pickle.load(open(args.model))
if args.gpu >= 0:
    model.to_gpu()

mean_image = numpy.load(open(args.mean))
tags = [line.rstrip() for line in open(args.tags)]
tag_dict = dict((i,tag) for i, tag in enumerate(tags))

img_preprocessed = (to_bgr(fetch_image(args.image_url)) - mean_image).transpose((2, 0, 1))

predicted = model.predict(xp.array([img_preprocessed]))[0]

top_10 = sorted(enumerate(predicted), key=lambda index_value: -index_value[1])[:30]
top_10_tag = [
    (tag_dict[key], float(value))
    for key, value in top_10 if value > 0
]
for tag, score in top_10_tag:
    print("tag: {} / score: {}".format(tag, score))
