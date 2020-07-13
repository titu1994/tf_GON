# Gradient Origin Network implementation for Tensorflow 2.0
Tensorflow 2.0 port of [Gradient Origin Networs](https://arxiv.org/abs/2007.02798) paper from the original author's implementation found at - https://github.com/cwkx/GON.

# Usage
## MNIST
MNIST models (4.3k parameter models) can be trained in under 500 steps to good quality, so they dont have weights restoration. They take just a few minutes to train, even on CPU.
Simply run `mnist_example.py`, and you it will create a directory called `mnist/images` which stores 3 kinds of images - reconstruction, sampled images and spherical interpolation between images.

## CIFAR 10
CIFAR 10 models are much larger - 260k parameters, with a larger latent space (256 dim), and need to be trained for a lot longer (100k train steps). Therefore weights have been provided along with log of training.

Simply run `cifar10_train.py` which takes around 12 - 14 hours on a good GPU to train.

To evaluate results with pre-trained model, run `cifar10_eval.py`. It will create images under `cifar10/images/*_eval.png`

<p align="center">
  <img src="https://github.com/titu1994/tf_GON/blob/master/images/CIFAR10_reconstruction.gif?raw=true">
</p>

# Requirements
`pip install -r requirements.txt`.

# Citation
Please cite the original authors for their work - 

```
@article{bondtaylor2020gradient,
  title={Gradient Origin Networks},
  author={Sam Bond-Taylor and Chris G. Willcocks},
  journal={arXiv preprint arXiv:2007.02798},
  year={2020}
}
```
