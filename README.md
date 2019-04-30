# noise2self-keras
A simple implementation of [noise2self](https://arxiv.org/abs/1901.11365), using keras/tensorflow backend in eager mode.

Reimplements some elements of [the official PyTorch implementation](https://github.com/czbiohub/noise2self). 

![An example of training noise2self on fashion-MNIST augmented with gaussian noise, after 16384 epochs ](https://raw.githubusercontent.com/zfisher/noise2self-keras/master/images/fashion-mnist.png)

## Usage

[Run a demo on Google Colab here.](https://drive.google.com/open?id=1jd5CBck3zuPfZ4HJP57LVWPyDL2H79C3)

To generate a basic example (removes Gaussian noise from MNIST):

```shell
python3 noise2self.py
````

Train on Fashion-MNIST with 500 batches:

```shell
python3 noise2self.py --dataset fashion-mnist --num-batches 500
````