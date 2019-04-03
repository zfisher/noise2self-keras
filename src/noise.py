import numpy as np
import tensorflow as tf

def add_gaussian_noise_np(x, min_std, max_std, clip = True):
    rng = np.random.uniform(min_std, max_std, [x.shape[0], 1, 1, 1])
    noise = np.random.normal(size=(1, *x.shape[1:])) * rng
    result = x + noise
    if clip:
        result = np.abs(result)
        result_inv = np.ones(result.shape) - result
        result = np.ones(result.shape) - np.abs(result_inv)
        result = np.clip(result, 0, 1)
    return result

def add_gaussian_noise_tf(x, min_std, max_std, clip = True):
    rng = tf.random_uniform(shape=[x.shape[0], 1, 1, 1], minval=min_std, maxval=max_std)
    noise = tf.random_normal(shape=(1, *x.shape[1:])) * rng
    result = x + noise
    if clip:
        result = tf.math.abs(result)
        result_inv = tf.ones(result.shape) - result
        result = tf.ones(result.shape) - tf.math.abs(result_inv)
        result = tf.clip_by_value(result, 0, 1)
    return result

def noisy_clean_generator(images, batch_size, min_std, max_std):
    while True:
        indices = np.random.choice(images.shape[0], batch_size)
        batch_clean = images[indices].reshape((batch_size, 28, 28, 1))
        batch_noise = add_gaussian_noise_np(batch_clean, min_std, max_std)
        yield batch_noise, batch_clean