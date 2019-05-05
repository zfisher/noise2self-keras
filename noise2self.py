import numpy as np
import os
import time
import argparse

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist, fashion_mnist
from tensorflow.contrib.eager.python import tfe

from src.models import BabyUnet
from src.noise import add_gaussian_noise_np, noisy_clean_generator
from src.utils import show_grid, show_plot, display_progress
from src.masker import Masker, infer

possible_datasets = ['mnist', 'fashion-mnist']
image_width, image_height = 28, 28

data_shape = (-1, image_width, image_height, 1)

def get_data(dataset):
    
    """ Prepare the mnist or fashion-mnist data to feed to the model.
    
    Args:
        dataset (string): one of the possible_datasets
    
    Returns:
        A tuple with the clean training data, clean testing data, and noisy
        testing data.
    """
    
    if dataset not in possible_datasets:
        datasets_output = ', '.join(possible_datasets)
        raise ValueError('dataset must be one of: {}'.format(datasets_output))
    
    if dataset == 'mnist':
        (clean_train, __), (clean_test, __) = mnist.load_data()
    elif dataset == 'fashion-mnist':
        (clean_train, __), (clean_test, __) = fashion_mnist.load_data()

    clean_train = clean_train.astype('float32') / 255.
    clean_train = clean_train.reshape(data_shape)
    clean_test = clean_test.astype('float32') / 255.
    clean_test = clean_test.reshape(data_shape)

    noisy_test  = add_gaussian_noise_np(clean_test, 0.0, 0.4)
    return clean_train, clean_test, noisy_test


def train_model(clean_train, clean_test, noisy_test, num_batches=150, 
                batch_size=32, show_loss_plot=False, verbose=False, seed=1337):
    
    """ Trains a UNet to learn denoising by self-supervision (noise2self).
    Uses matplotlib to display the results.

    Args:
        clean_train (tensor): the clean training data
        clean_test (tensor): the clean testing data
        clean_test (tensor): the noisy testing data
        num_batches (int): number of batches used for training
        batch_size (int): number of images in each batch
        show_loss_plot (bool): display a graph of loss after training
        verbose (bool): print extra information
        seed (int): random seed for tensorflow and numpy

    Returns:
        The trained tensorflow model.
    """
    
    if num_batches <= 0:
        raise ValueError('must have a positive number of batches')
    if batch_size <= 0:
        raise ValueError('must have a positive batch size')
    
    def verbose_print(s):
        if verbose:
            print(s)
    
    tf.enable_eager_execution()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    device = '/gpu:0' if tfe.num_gpus() else '/cpu:0'

    with tf.device(device):
        verbose_print('building model (device={})'.format(device))
        model = BabyUnet()
        model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                      loss=tf.keras.losses.mean_squared_error)
        model.build((1, image_width, image_height ,1))
    
        optimizer = tf.train.AdamOptimizer()
        loss_fn = tf.losses.mean_squared_error
        loss_history = []
    
        noise_gen = noisy_clean_generator(clean_train, batch_size, 0, 0.4)
        masker = Masker(interpolate=True, spacing=4, radius=1)
    
        verbose_print('fitting model')
        start_time = time.time()
        loss_display = ''
        for (batch, (batch_noisy, batch_clean)) in enumerate(noise_gen):
            fraction_display = '{}/{}'.format(batch, num_batches).rjust(10)
            display_progress(batch, num_batches, length=30, suffix=loss_display,
                             prefix=fraction_display)
            
            if batch == num_batches:
                break
        
            with tf.GradientTape() as tape:
                masked, mask = masker(batch_noisy, batch, shape=data_shape)
                batch_predictions = model(tf.cast(masked, tf.float32))
                loss_value = loss_fn(mask * batch_clean, mask * batch_predictions)
        
            loss_display = '(loss: {:0.6f})'.format(loss_value.numpy())
            loss_history.append(loss_value.numpy())
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables),
                global_step=tf.train.get_or_create_global_step()
            )
        
        end_time = time.time()
        verbose_print('fit completed in {:0.2f}s'.format(end_time - start_time))
    
        if show_loss_plot:
            show_plot(loss_history, 'Loss', 'Epoch', 'Mean Square Error Loss')
        
        verbose_print('validating')
        masked, mask = masker(noisy_test, 0, shape=data_shape)
        test_predictions = model(tf.cast(masked, tf.float32))
        test_loss_value = loss_fn(mask * clean_test, mask * test_predictions)
        
        print("final test loss: {:0.6f}".format(test_loss_value))
    
    return model

def save_model_weights(model, output_path):
    """ Saves the weights of model to output_path.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.save_weights(output_path)

def plot_examples(model, clean_test, noisy_test, num_examples=15, 
                  randomize=False, output_path=None):
    
    """ Generates a set of examples from the trained model.
    
    Args:
        model (tensorflow model): the trained model
        clean_test (tensor): the clean testing data
        noisy_test (tensor): the noisy testing data
        num_examples (int): how many examples to show in pyplot
        randomize (bool): if False, use the first num_examples images
        output_path (string): path to output figure, or None for standard

    Returns:
        None.
    
    """
    
    if num_examples <= 0:
        raise ValueError('must generate a positive number of examples')
    
    if randomize:
        indices = np.random.choice(clean_test.shape[0], num_examples)
    else:
        indices = list(range(num_examples))
    
    cleans = tf.reshape(clean_test[indices], data_shape)
    noisys = tf.reshape(noisy_test[indices], data_shape)
    predictions = model.predict(noisy_test[indices])
    inferences = infer(noisys, model, spacing=4)

    titles = ['ground truth', 'augmented with gaussian noise',
              'neural network output', 'J-invariant reconstruction']
    show_grid([cleans, noisys, predictions, inferences], titles=titles,
              output_path=output_path)


if __name__ == '__main__':
    program_desc = 'Generate some examples of denoising via self-supervision'\
                   ' with noise2self.'
    parser = argparse.ArgumentParser(description=program_desc)
    
    parser.add_argument('--dataset', dest='dataset', type=str, default='mnist',
                        help='either mnist or fashion-mnist')
    parser.add_argument('--num-batches', dest='num_batches', type=int,
                        default=150, help='number of batches')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=32, help='batch size')
    parser.add_argument('--num-examples', dest='num_examples', type=int,
                        default=15, help='number of examples to plot')
    parser.add_argument('--show-loss-plot', dest='show_loss', 
                        action='store_true', help='display a plot with losses')
    parser.add_argument('--weight-output-path', dest='weight_output_path', 
                        type=str, default=None, 
                        help='path to output the weights file (in hdf5 format)')
    parser.add_argument('--example-output-path', dest='example_output_path', 
                        type=str, default=None, 
                        help='path to output example figure')
    parser.add_argument('-v', '--verbose', dest='verbose', 
                        action='store_true', help='verbose mode')
    
    args = parser.parse_args()
    
    clean_train, clean_test, noisy_test = get_data(args.dataset)
    model = train_model(clean_train, clean_test, noisy_test, args.num_batches, 
                        args.batch_size, args.show_loss, args.verbose)
    
    if args.num_examples:
        plot_examples(model, clean_test, noisy_test, args.num_examples, 
                      args.example_output_path)
    
    if args.weight_output_path:
        save_model_weights(model, args.weight_output_path)