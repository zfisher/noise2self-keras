# noise2self-keras
A simple implementation of [noise2self](https://arxiv.org/abs/1901.11365), using keras/tensorflow backend in eager mode.

Reimplements some elements of [the official PyTorch implementation](https://github.com/czbiohub/noise2self). 

![An example of training noise2self on MNIST augmented with gaussian noise, after 150 epochs ](https://raw.githubusercontent.com/zfisher/noise2self-keras/master/images/mnist.png)

## Usage

For a demo (removes Gaussian noise from MNIST):

```shell
python3 noise2self.py
````

Train on Fashion-MNIST with 500 batches:

```shell
python3 noise2self.py --dataset fashion-mnist --num-batches 500
````