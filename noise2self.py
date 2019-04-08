import numpy as np
import os
import sys
import time
import argparse

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist, fashion_mnist
from tensorflow.contrib.eager.python import tfe

sys.path.insert(0, os.path.join(sys.path[0], 'src'))

from models import BabyUnet
from noise import add_gaussian_noise_np, noisy_clean_generator
from utils import show_grid, show_plot, display_progress
from masker import Masker, infer

possible_datasets = ['mnist', 'fashion-mnist']
image_width, image_height = 28, 28

def generate_examples(dataset, num_batches, batch_size, num_examples, 
                      show_loss_plot, output_path, verbose):
    assert num_batches > 0, \
        'must have a positive number of batches'
    assert batch_size > 0, \
        'must have a positive batch size'
    assert num_examples > 0, \
        'must have a positive number of examples'
    assert dataset in possible_datasets, \
        'dataset must be one of: {}'.format(', '.join(possible_datasets))
    
    tf.enable_eager_execution()

    tf.set_random_seed(int(time.time()))
    np.random.seed(int(time.time()))
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    if dataset == 'mnist':
        (clean_train, __), (clean_test, __) = mnist.load_data()
    elif dataset == 'fashion-mnist':
        (clean_train, __), (clean_test, __) = fashion_mnist.load_data()

    data_shape = (-1, image_width, image_height, 1)
    clean_train = clean_train.astype('float32') / 255.
    clean_train = clean_train.reshape(data_shape)
    clean_test = clean_test.astype('float32') / 255.
    clean_test = clean_test.reshape(data_shape)

    noisy_test  = add_gaussian_noise_np(clean_test, 0.0, 0.4)

    loss_fn = tf.losses.mean_squared_error

    device = '/gpu:0' if tfe.num_gpus() else '/cpu:0'

    with tf.device(device):
        if verbose:
            print('building model (device={})'.format(device))
        model = BabyUnet()
        dummy_x = tf.zeros((1, image_width, image_height, 1))
        model._set_inputs(dummy_x)
        model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                      loss=tf.keras.losses.mean_squared_error)
        model.build((1, image_width, image_height ,1))
    
        optimizer = tf.train.AdamOptimizer()
        masker = Masker(interpolate=True, spacing=4, radius=1)
        loss_history = []
    
        noise_gen = noisy_clean_generator(clean_train, batch_size, 0, 0.4)
    
        if verbose:
            print('fitting model')
        start_time = time.time()
        last_loss = ''
        for (batch, (batch_noisy, batch_clean)) in enumerate(noise_gen):
            display_progress(batch, num_batches, length=30, suffix=last_loss,
                             prefix='{}/{}'.format(batch, num_batches).rjust(10))
            if batch == num_batches:
                break
        
            with tf.GradientTape() as tape:
                masked, mask = masker(batch_noisy, batch)
                masked = tf.reshape(masked, data_shape)
                mask   = tf.reshape(mask,   data_shape)
                batch_predictions = model(tf.cast(masked, tf.float32))
                loss_value = loss_fn(mask * batch_clean, mask * batch_predictions)
        
            last_loss = '(loss: {:0.5f})'.format(loss_value.numpy())
            loss_history.append(loss_value.numpy())
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())
        end_time = time.time()
        if verbose:
            print('fit completed in {:0.2f}s'.format(end_time - start_time))
    
        if show_loss_plot:
            show_plot(loss_history, 'Loss', 'Epoch', 'Mean Square Error Loss')
        
        if verbose:
            print('validating')

        scores = model.evaluate(noisy_test, clean_test, 32)
        print("final test loss: {:0.3f}".format(scores))
    
        # prepare some example output
        indices = np.random.choice(clean_test.shape[0], num_examples)
        cleans = tf.reshape(clean_test[indices], data_shape)
        noisys = tf.reshape(noisy_test[indices], data_shape)
        preds = model.predict(noisy_test[indices])
    
        maskeds, masks = masker(noisys, 0)
        maskeds = tf.reshape(maskeds, (num_examples, image_width, image_height))
    
        infs = infer(noisys, model, spacing=4)
    
        titles = ['ground truth', 'augmented with gaussian noise',
                  'neural network output', 'J-invariant reconstruction']
        show_grid([cleans, noisys, preds, infs], titles=titles)
    
        model.save_weights(output_path)

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
    parser.add_argument('-v', '--verbose', dest='verbose', 
                        action='store_true', help='verbose mode')
    parser.add_argument('--output-path', dest='output_path', type=str, 
                        default='weights/baby_unet.h5', 
                        help='path to output the weights file (in hdf5 format)')
    
    args = parser.parse_args()
    
    generate_examples(args.dataset, args.num_batches, args.batch_size, 
                      args.num_examples, args.show_loss, args.output_path, 
                      args.verbose)