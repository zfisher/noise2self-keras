num_batches = 300
num_examples = 15
show_loss_plot = True

import numpy as np
import os
import time

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.contrib.eager.python import tfe

from models import BabyUnet
from noise import add_gaussian_noise_np, noisy_clean_generator
from utils import show_grid, show_plot, display_progress
from masker import Masker, infer

tf.enable_eager_execution()

tf.set_random_seed(int(time.time()))
np.random.seed(int(time.time()))

if not os.path.exists('weights/mnist/'):
    os.makedirs('weights/mnist/')

(clean_train, __), (clean_test, __) = mnist.load_data()

clean_train = clean_train.astype('float32') / 255.
clean_train = clean_train.reshape((-1, 28, 28, 1))
clean_test = clean_test.astype('float32') / 255.
clean_test = clean_test.reshape((-1, 28, 28, 1))

noisy_train = add_gaussian_noise_np(clean_train, 0, 0.4)
noisy_test  = add_gaussian_noise_np(clean_test, 0, 0.4)

loss_fn = tf.losses.mean_squared_error

device = '/gpu:0' if tfe.num_gpus() else '/cpu:0'

with tf.device(device):
    print('building model (device={})'.format(device))
    model = BabyUnet()
    print('compiling model')
    dummy_x = tf.zeros((1, 28, 28, 1))
    model._set_inputs(dummy_x)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss=tf.keras.losses.mean_squared_error)
    model.build((1,28,28,1))
    
    print('fitting')
    optimizer = tf.train.AdamOptimizer()
    masker = Masker(spacing=3)
    loss_history = []
    
    noise_gen = noisy_clean_generator(clean_train, 32, 0, 0.4)

    print()
    start_time = time.time()
    for (batch, (batch_noisy, batch_clean)) in enumerate(noise_gen):
        display_progress(batch, num_batches)
        if batch == num_batches:
            break
        
        with tf.GradientTape() as tape:
            masked, mask = masker(tf.cast(batch_noisy,tf.float64), batch)
            masked = tf.reshape(masked, (-1,28,28,1))
            mask = tf.reshape(masked, (-1,28,28,1))
            batch_predictions = model(tf.cast(masked,tf.float32))
            loss_value = loss_fn(mask*batch_clean, mask*batch_predictions)
        
        loss_history.append(loss_value.numpy())
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step=tf.train.get_or_create_global_step())
    end_time = time.time()
    print('fit completed in {:0.2f}s'.format(end_time - start_time))
    
    if show_loss_plot:
        show_plot(loss_history, 'Loss', 'Epoch', 'Mean Square Error Loss')
    
    print('validating')
    scores = model.evaluate(noisy_test, clean_test, 32)
    print("final test loss: ", round(scores, 3))
    
    # prepare some example output
    indices = np.random.choice(clean_test.shape[0], num_examples)
    cleans = tf.reshape(clean_test[indices], (num_examples,28,28, 1))
    noisys = tf.reshape(noisy_test[indices], (num_examples, 28, 28, 1))
    preds = model.predict(noisy_test[indices])
    preds = np.clip(preds, 0, 1)
    maskeds, masks = masker(noisys, 0)
    maskeds = tf.reshape(maskeds,(num_examples, 28, 28))
    
    infs = infer(noisys, model, 3)
    infs = tf.reshape(infs,(num_examples, 28, 28))
    
    # clip the output before pyplot.
    infs = np.clip(infs, 0, 1)
    maskeds = np.clip(maskeds,0,1)
    
    titles = ['ground truth', 'augmented with gaussian noise',
              'neural network output', 'masked noisy image', 
              'J-invariant reconstruction']
    show_grid([cleans, noisys, preds, maskeds, infs], titles=titles)
    
    model.save_weights('weights/mnist/baby_unet.h5')