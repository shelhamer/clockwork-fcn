import os
import glob
import operator
import re
import numpy as np
import scipy.io as sio

from datetime import datetime, timedelta
from PIL import Image


class nyud:
    def __init__(self, data_path):
        # data_path something like /data4/nyud
        self.dir = data_path
        self.seg_dir = os.path.join(data_path, 'segmentation')
        self.raw_dir = os.path.join(data_path, 'raw')
        self.num_classes = 40
        self.mean = np.array((116.2, 97.2, 92.3), dtype=np.float32)

    def get_dset(self, raw=False):
        '''
        RAW dataset will be returned as a list of tuples (video, frame)
        SEG dataset will be returned as a list of identifiers
        '''
        if raw:
            ls = open('{}/test_rawImages.txt'.format(self.dir)).read().splitlines()
            return [(item.split('/')[0], item.split('/')[1][:-4])
                    for item in ls]
        else:
            return open('{}/test.txt'.format(self.dir)).read().splitlines()

    def get_video(self, img):
        return img.split('/')[0]

    def get_frame(self, img):
        return img.split('/')[1][:-4]

    def get_datetime(self, fn):
        '''
        Extract the matlabtime from the filename as demonstrated in
        "get_timestamp_from_filename.m" in toolbox of NYU dataset
        '''
        ms_time = float(re.search('-([0-9,.]*)-', fn).group(1))
        return ms_time

    def list_frames_timestamps(self, vid):
        '''
        List all the frames in the video sequentially as (frame, timestamp)
        '''
        # list *all* frames
        frames = [os.path.basename(f)[:-4]
                  for f in glob.glob('{}/{}/*.ppm'.format(self.raw_dir, vid))]
        timestamps = [(f, self.get_datetime(f)) for f in frames]
        timestamps.sort(key=operator.itemgetter(1))  # order based on timestamp
        return timestamps

    def list_frames(self, vid):
        timestamps = self.list_frames_timestamps(vid)
        return [t[0] for t in timestamps]

    def get_val2raw_frames(self):
        '''
        Return the valset as a list of (idx, raw frame)
        '''
        valset = self.get_dset('val')
        rawset = self.get_dset('raw')
        raw_labeled = []
        for frm, tup in zip(valset, rawset):
            vid, gt_frame = tup[0], tup[1]
            timestamps = self.list_frames_timestamps(vid)
            # find our ground truth frame
            fidx = timestamps.index((gt_frame, self.get_datetime(gt_frame)))
            raw_labeled += [(frm, timestamps[fidx][0])]
        return raw_labeled

    def list_subsampled_frames(self, vid, gt_frame, skip):
        '''
        Get subsampled video frames at rate "skip"
        If the video is not long enough, return the empty list
        Used for adaptive clockwork processing
        '''
        timestamps = self.list_frames_timestamps(vid)
        # find our ground truth frame
        fidx = timestamps.index((gt_frame, self.get_datetime(gt_frame)))
        keepers = [item[0] for item in timestamps[fidx::-1*skip]][1:] # grab subsampled frames preceding labeled frame
        keepers.reverse()
        keepers += [item[0] for item in timestamps[fidx::skip]] # grab subsampled frames following labeled frame
        assert len(set(keepers)) == len(keepers) # only have one of each frame
        return keepers
        #return [(vid, k) for k in keepers]

    def list_preceding_frames(self, skip, vid, gt_frame):
        '''
        Get "num_frames" frames preceding "gt_frame" at sample rate "skip"
        Returns empty list when not enough frames to generate the list
        Used for fixed rate clockwork and pipeline experiments
        '''
        timestamps = self.list_frames_timestamps(vid)
        # find our ground truth frame
        fidx = timestamps.index((gt_frame, self.get_datetime(gt_frame)))
        keepers = [item[0] for item in timestamps[fidx::-1*skip]] # grab num_frames subsampled frames preceding labeled frame
        keepers.reverse()
        return [(vid, k) for k in keepers]

    def load_raw_image(self, vid, idx):
        """
        Load input image from raw dataset as PIL image
        """
        im = Image.open('{}/{}/{}.ppm'.format(self.raw_dir, vid, idx))
        return im

    def load_seg_image(self, idx):
        '''
        Load test image (has GT) as PIL image
        '''
        im = Image.open('{}/images/img_{}.png'.format(self.dir, idx))
        return im

    def preprocess(self, im):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices,
        return as numpy array.
        """
        label = sio.loadmat('{}/img_{}.mat'.format(self.seg_dir, idx))['segmentation'].astype(np.uint8)
        label -= 1  # shift labels down so ignore label == 255
        return label

    def crop_frame(self, frame):
        '''
        Center crop the raw frame to be the same size as the label image
        Modeled after 'crop_image.m' from the NYUD V2 Toolbox
        Saurabh CropIt: 46:470, 41:600
        '''
        frame = np.array(frame, dtype=np.uint8)
        left = 40
        top = 44
        right = 600
        bottom = 469
        return frame[top:bottom, left:right]
