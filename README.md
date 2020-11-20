# CACTUs-ProtoNets
CACTUs-ProtoNets: Clustering to Automatically Generate Tasks for Unsupervised Prototypical Networks.

This code was used to produce the CACTUs-Protonets results and baselines in the paper [Unsupervised Learning via Meta-Learning](https://arxiv.org/abs/1810.02334).

This repository was built off of [Prototypical Networks for Few-Shot Learning](https://github.com/jakesnell/prototypical-networks).

### Dependencies
The code was tested with the following setup:
* Ubuntu 16.04
* Python 3.6.6
* PyTorch 0.4.0

Instructions:
* Install [PyTorch and torchvision](http://pytorch.org/).
* Install [torchnet](https://github.com/pytorch/tnt) by running `pip install git+https://github.com/pytorch/tnt.git@master`.
* Install the protonets package by running `python setup.py install` or `python setup.py develop`.
* Install scikit-learn.


### Data
The Omniglot splits with ACAI and BiGAN encodings used for the results in the paper are available [here](https://drive.google.com/file/d/1i6kEbySnR51jT3pW_60E3PGkIOKmxTfQ/view).
Download and extract the archive's contents into this directory.

Unfortunately, due to licensing issues, I am not at liberty to re-distribute the miniImageNet or CelebA datasets. The code for these datasets is still presented for posterity.

### Usage
You can find examples of scripts in ```/scripts```. All results were obtained using a single GPU.

### Credits
The unsupervised representations were computed using three open-source codebases from prior works.

* [Adversarial Feature Learning](https://github.com/jeffdonahue/bigan)
* [Deep Clustering for Unsupervised Learning of Visual Features](https://github.com/facebookresearch/deepcluster)
* [Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer](https://github.com/brain-research/acai)

### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/hsukyle/cactus-maml/issues).

