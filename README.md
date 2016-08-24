# Clockwork Convnets for Video Semantic Segmentation

This is the reference implementation of [arxiv:1608.03609](https://arxiv.org/abs/1608.03609):

    Clockwork Convnets for Video Semantic Segmentation
    Evan Shelhamer*, Kate Rakelly*, Judy Hoffman*, Trevor Darrell
    arXiv:1605.06211

This project reproduces results from the arxiv and demonstrates how to execute staged fully convolutional networks (FCNs) on video in [Caffe](https://github.com/BVLC/caffe) by controlling the net through the Python interface.
In this way this these experiments are a proof-of-concept implementation of clockwork, and further development is needed to achieve peak efficiency (such as pre-fetching video data layers, threshold GPU layers, and a native Caffe library edition of the staged forward pass for pipelining).

For simple reference, refer to these (display only) editions of the experiments:

- [Cityscapes Clockwork](http://nbviewer.jupyter.org/github/shelhamer/clockwork-fcn/blob/master/notebooks/cityscapes-clockwork-exp.ipynb)
- [YouTube Frame Differencing](http://nbviewer.jupyter.org/github/shelhamer/clockwork-fcn/blob/master/notebooks/youtube-differences.ipynb)
- [YouTube Clockwork](http://nbviewer.jupyter.org/github/shelhamer/clockwork-fcn/blob/master/notebooks/youtube-clockwork-exp.ipynb)
- [YouTube Pipelining](http://nbviewer.jupyter.org/github/shelhamer/clockwork-fcn/blob/master/notebooks/youtube-pipeline-exp.ipynb)
- [Synthetic PASCAL VOC Video](http://nbviewer.jupyter.org/github/shelhamer/clockwork-fcn/blob/master/notebooks/pascal-translate-exp.ipynb)
- Dataset Walkthroughs for [YouTube](http://nbviewer.jupyter.org/github/shelhamer/clockwork-fcn/blob/master/notebooks/data-youtube.ipynb), [NYUDv2](http://nbviewer.jupyter.org/github/shelhamer/clockwork-fcn/blob/master/notebooks/data-nyud.ipynb), and [Cityscapes](http://nbviewer.jupyter.org/github/shelhamer/clockwork-fcn/blob/master/notebooks/data-cityscapes.ipynb)

**Contents**

- `notebooks`: interactive code and documentation that carries out the experiments (in jupyter/ipython format).
- `nets`: the net specification of the various FCNs in this work, and the pre-trained weights (see installation instructions).
- `caffe`: the Caffe framework, included as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) pointing to a compatible version
- `datasets`: input-output for PASCAL VOC, NYUDv2, YouTube-Objects, and Cityscapes
- `lib`: helpers for executing networks, scoring metrics, and plotting

## License

This project is licensed for open non-commercial distribution under the UC Regents license; see [LICENSE](./LICENSE).
Its dependencies, such as Caffe, are subject to their own respective licenses.

## Requirements & Installation

Caffe, Python, and Jupyter are necessary for all of the experiments.
Any installation or general Caffe inquiries should be directed to the [caffe-users](mailto:caffe-users@googlegroups.com) mailing list.

1. Install Caffe. See the [installation guide](http://caffe.berkeleyvision.org/installation.html) and try [Caffe through Docker](https://github.com/BVLC/caffe/tree/master/docker) (recommended).
*Make sure to configure pycaffe, the Caffe Python interface, too.*
2. Install Python, and then install our required packages listed in  `requirements.txt`.
For instance, `for x in $(cat requirements.txt); do pip install $x; done` should do.
3. Install [Jupyter](http://jupyter.org/), the interface for viewing, executing, and altering the notebooks.
4. Configure your `PYTHONPATH` as indicated by the included `.envrc` so that this project dir and pycaffe are included.
5. Download the [model weights](http://dl.caffe.berkeleyvision.org/clockwork-nets.tar.gz) for this project and place them in `nets`.

Now you can explore the notebooks by firing up Jupyter.
