import sys
import numpy as np
import random
from PIL import Image

import caffe


class CityscapesSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) fine annotation pairs from Cityscapes
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.

    N.B. Only half and image is loaded at a time due to memory constraints, but
    care is taken to guarantee equivalence to whole image processing. Every
    crop must be processed for this equivalence to hold, effectively making the
    training + val sets twice as large for indexing and the like.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - dir: path to Cityscapes dir
        - split: train/val/trainval/test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for Cityscapes semantic segmentation.

        example

        params = dict(dir='/path/to/Cityscapes', split='val')
        """
        # config
        params = eval(self.param_str)
        self.dir = params['cscapes_dir']
        self.split = params['split']
        self.mean = np.array((72.78044, 83.21195, 73.45286), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # import cityscapes label helper and set up label mappings
        sys.path.insert(0, '{}/scripts/helpers/'.format(self.dir))
        labels = __import__('labels')
        self.id2trainId = {label.id: label.trainId for label in labels.labels}  # dictionary mapping from raw IDs to train IDs

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.indices = []
        for s in self.split:
            split_f = '{}/ImageSets/segFine/{}.txt'.format(self.dir, s)
            self.indices.extend(self.prepare_input(open(split_f, 'r').read().splitlines(), s))
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            random.shuffle(self.indices)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(*self.indices[self.idx])
        self.label = self.load_label(*self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        self.idx += 1
        if self.idx == len(self.indices):
            if self.random:
                random.shuffle(self.indices)
            self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def prepare_input(self, indices, split):
        """
        Augment each index with left/right pair and its split for loading
        half-image crops to cope with memory limits.
        """
        full_indices = [(idx, split, 'right') for idx in indices]
        full_indices.extend([(idx, split, 'left') for idx in indices])
        return full_indices

    def half_crop_image(self, im, position, label=False):
        """
        Generate a crop of full height and width = width/2 + overlap.
        Align the crop along the left or right border as specified by position.
        If the image is a label, ignore the pixels in the overlap.
        """
        overlap = 210
        w = im.shape[1]
        if position == 'left':
            crop = im[:, :(w / 2 + overlap)]
            if label:
                crop[:, (w / 2):(w / 2 + overlap)] = 255
        elif position == 'right':
            crop = im[:, (w/2 - overlap):]
            if label:
                crop[:, :overlap] = 255
        else:
            raise Exception("Unsupported crop")
        return crop

    def load_image(self, idx, split, position):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        full_im = np.array(Image.open('{}/images/leftImg8bit/{}/{}_leftImg8bit.png'.format(self.dir, split, idx)), dtype=np.uint8)
        im = self.half_crop_image(full_im, position, label=False)
        in_ = im.astype(np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    def assign_trainIds(self, label):
        """
        Map the given label IDs to the train IDs appropriate for training
        This will map all the classes we don't care about to label 255
        Use the label mapping provided in labels.py
        """
        for k, v in self.id2trainId.iteritems():
            label[label == k] = v
        return label

    def load_label(self, idx, split, position):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        full_label = np.array(Image.open('{}/trainvaltest/gtFine/{}/{}_gtFine_labelIds.png'.format(self.dir, split, idx)), dtype=np.uint8)
        label = self.half_crop_image(full_label, position, label=True)
        label = self.assign_trainIds(label)
        label = np.array(label, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label
