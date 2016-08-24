import os
import copy
import glob
import numpy as np

from PIL import Image


class pascal:
    def __init__(self, data_path):
        # data_path something like /x/pascal/VOC2011
        self.dir = data_path
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.mean = (104.00698793, 116.66876762, 122.67891434)  # imagenet mean
        # for paletting
        reference_idx = '2008_000666'
        palette_im = Image.open('{}/SegmentationClass/{}.png'.format(
            self.dir, reference_idx))
        self.voc_palette = palette_im.palette

    def get_dset(self):
        """
        Load seg11valid, the non-intersecting set of PASCAL VOC 2011 segval
        and SBD train.
        """
        segset_dir = '{}/ImageSets/Segmentation'.format(self.dir)
        return open('{}/seg11valid.txt'.format(segset_dir)).read().splitlines()

    def load_image(self, idx):
        im = Image.open('{}/JPEGImages/{}.jpg'.format(self.dir, idx))
        return im

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        label = Image.open('{}/SegmentationClass/{}.png'.format(self.dir, idx))
        label = np.array(label, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label

    def preprocess(self, im):
        """
        Preprocess loaded image (by load_image) for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= np.array(self.mean)
        in_ = in_.transpose((2, 0, 1))
        return in_

    def palette(self, label_im):
        '''
        Transfer the VOC color palette to an output mask
        '''
        if label_im.ndim == 3:
            label_im = label_im[0]
        label = Image.fromarray(label_im, mode='P')
        label.palette = copy.copy(self.voc_palette)
        return label

    def make_translated_frames(self, im, label, shift=None, num_frames=None):
        """
        Extract corresponding image and label crops.
        Shift by `shift` pixels at a time for a total of `num_frames`
        so that the total translation is `shift * (num_frames - 1)`.
        im should be prepared by preprocess_voc and gt by load_gt
        """
        assert(shift is not None and num_frames is not None)
        im = np.asarray(im)
        im_crops = []
        label_crops = []
        # find largest dimension, fit crop to shift and frames
        max_dim, shift_idx = np.max(im.shape), np.argmax(im.shape)
        crop_dim = max_dim - shift * (num_frames - 1)
        crop_shape = list(im.shape)
        crop_shape[shift_idx] = crop_dim
        # determine shifts
        crop_shifts = np.arange(0, max_dim - crop_dim + 1, shift)
        for sh in crop_shifts:
            # TODO(shelhamer) there has to be a better way
            crop_idx = [slice(None)] * 3
            crop_idx[shift_idx] = slice(sh, sh + crop_dim)
            im_crops.append(im[crop_idx])
            label_crops.append(label[[0] + crop_idx[:-1]])
        # output is (# frames, channels, spatial)
        im_crops = np.asarray(im_crops)
        label_crops = np.asarray(label_crops)[:, np.newaxis, :, :]
        return im_crops, label_crops

    def make_boundaries(self, label, thickness=None):
        """
        Input is an image label, output is a numpy array mask encoding the
        boundaries of the objects
        Extract pixels at the true boundary by dilation - erosion of label.
        Don't just pick the void label: it is not exclusive to the boundaries.
        """
        assert(thickness is not None)
        import skimage.morphology as skm
        void = 255
        mask = np.logical_and(label > 0, label != void)[0]
        selem = skm.disk(thickness)
        boundaries = np.logical_xor(skm.dilation(mask, selem),
                                    skm.erosion(mask, selem))
        return boundaries
