import sys
sys.path.append('python/')

import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from caffe.coord_map import crop

# helper function for common structures

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fcn(split, num_classes=None):
    n = caffe.NetSpec()
    n.data = L.Input(shape=[dict(dim=[1, 3, 500, 500])])

    # the net itself
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)

    # FCN edition
    n.score_fr = L.Convolution(n.pool4, num_output=num_classes, kernel_size=1,
        pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    n.upscore = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=num_classes, kernel_size=32, stride=16,
            bias_term=False),
        param=[dict(lr_mult=0)])
    n.score = crop(n.upscore, n.data)

    return n.to_proto()

def make_net():
    with open('voc-fcn-pool4.prototxt', 'w') as f:
        f.write(str(fcn('deploy', 21)))
    with open('nyud-fcn-pool4.prototxt', 'w') as f:
        f.write(str(fcn('deploy', 40)))
    with open('cityscapes-fcn-pool4.prototxt', 'w') as f:
        f.write(str(fcn('deploy', 20)))  # TODO should be 19

if __name__ == '__main__':
    make_net()
