import numpy as np

def get_out_scoremap(net):
    return net.blobs['score'].data[0].argmax(axis=0).astype(np.uint8)

def feed_net(net, in_):
    """
    Load prepared input into net.
    """
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

def segrun(net, in_):
    feed_net(net, in_)
    net.forward()
    return get_out_scoremap(net)

def pipeline_fill_3stage(net, in0, in1):
    feed_net(net, in0)
    net.forward()
    pipeline_2stage_forward(net, in1)

def pipeline_fill_2stage(net, in_):
    feed_net(net, in_)
    net.forward()

def pipeline_2stage_forward(net, in_):
    feed_net(net, in_)
    net.forward(start='conv5_1', end='upscore2')
    net.forward(start='conv1_1', end='score_pool4')
    net.forward(start='score_pool4c')
    return get_out_scoremap(net)

def pipeline_3stage_forward(net, in_):
    feed_net(net, in_)
    net.forward(start='conv5_1', end='upscore2')
    net.forward(start='conv4_1', end='score_pool4')
    net.forward(start='conv1_1', end='score_pool3')
    net.forward(start='score_pool4c')
    return get_out_scoremap(net)

def clockwork_forward(net, in_):
    feed_net(net, in_)
    net.forward(start='conv1_1', end='score_pool4')
    net.forward(start='score_pool4c')
    return get_out_scoremap(net)

def adaptive_clock_forward(net, in_, clock):
    if clock:
        out = segrun(net, in_)
    else:
        out = clockwork_forward(net, in_)
    return out
