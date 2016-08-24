import os
import glob
import numpy as np
import PIL
from PIL import Image


class youtube:
    def __init__(self, data_path):
        self.dir = data_path
        self.classes = ['background', 'aeroplane', 'bird', 'boat',
                        'car', 'cat', 'cow', 'dog', 'horse', 'motorbike',
                        'train']
        # same mean as PASCAL VOC (the imagenet mean) for compatibility w/ FCN
        self.mean = (104.00698793, 116.66876762, 122.67891434)
        self.MAX_DIM = 500.0  # match PASCAL VOC training data
        self.label_thresh = 15

    def list_vids(self, class_):
        """Returns the ids for the videos in which class_ appears."""
        files = [os.path.basename(f) for f in glob.glob('{}/v1/{}/data/*'.format(self.dir, class_))]
        dirs = filter(lambda x: os.path.isdir('{}/v1/{}/data/{}'.format(self.dir, class_, x)), files)
        return dirs

    def list_shots(self, class_, vid):
        """All shots which contain class_ from video vid."""
        # TODO: verify that the description is correct
        return [os.path.basename(f) for f in glob.glob('{}/v1/{}/data/{}/shots/*'.format(self.dir, class_, vid))]

    def list_frames(self, class_, vid, shot):
        """List the frames for class_ video vid and particular shot"""
        frames = [f.split('/')[-1].split('.')[0] for f in glob.glob('{}/v1/{}/data/{}/shots/{}/*.jpg'.format(self.dir, class_, vid, shot))]
        frames = [int(f[5:]) for f in frames]
        frames = sorted(frames)
        return frames

    def list_label_vids(self, class_):
        files = [os.path.basename(f) for f in glob.glob('{}/youtube_masks/{}/data/*'.format(self.dir, class_))]
        dirs = filter(lambda x: os.path.isdir('{}/youtube_masks/{}/data/{}'.format(self.dir, class_, x)), files)
        return dirs

    def list_label_shots(self, class_, vid):
        return [os.path.basename(f) for f in glob.glob('{}/youtube_masks/{}/data/{}/shots/*'.format(self.dir, class_, vid))]

    def list_label_frames(self, class_, vid, shot):
        fmt = '{}/youtube_masks/{}/data/{}/shots/{}/labels/*.jpg'.format(self.dir, class_, vid, shot)
        frames = [f.split('/')[-1].split('.')[0] for f in glob.glob(fmt)]
        frames = sorted([int(f) for f in frames])
        return frames

    def load_frame(self, class_, vid, shot, idx):
        im = Image.open('{}/v1/{}/data/{}/shots/{}/frame{:0>4d}.jpg'.format(self.dir, class_, vid, shot, int(idx)))
        im = self.resize(im, label=False)
        return np.array(im)

    def load_label(self, class_, vid, shot, idx):
        label = Image.open('{}/youtube_masks/{}/data/{}/shots/{}/labels/{:0>5d}.jpg'.format(self.dir, class_, vid, shot, int(idx)))
        label = self.resize(label, label=True)
        return label

    def resize(self, im, label=False):
        dims = np.array(im).shape
        if len(dims) > 2:
            dims = dims[:-1]
        max_val, max_idx = np.max(dims), np.argmax(dims)
        scale = self.MAX_DIM / max_val
        new_height, new_width = int(dims[0]*scale), int(dims[1]*scale)
        if label:
            im = im.resize((new_width, new_height), resample=PIL.Image.NEAREST)
        else:
            im = im.resize((new_width, new_height), resample=PIL.Image.BILINEAR)
        return im

    def convert_yt2voc_label(self, label, class_, voc_classes):
        label = np.array(label, dtype=np.uint8)
        label = label[np.newaxis, ...]
        label[label <= self.label_thresh] = 0
        label[label > self.label_thresh] = voc_classes.index(class_)
        return label

    def make_label(self, label, class_):
        label = np.array(label, dtype=np.uint8)
        label = label[np.newaxis, ...]
        label[label <= self.label_thresh] = 0
        label[label > self.label_thresh] = self.classes.index(class_)
        return label

    def preprocess(self, im):
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= np.array(self.mean)
        in_ = in_.transpose((2, 0, 1))
        return in_

    def load_dataset(self):
        """
        List all (class, video, shot) indices in the dataset.
        """
        indices = []
        for c in self.classes:
            for v in self.list_label_vids(c):
                for s in self.list_label_shots(c, v):
                    indices.append((c, v, s))
        return indices
