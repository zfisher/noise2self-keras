import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def infer(x, model, spacing):
    masker = Masker(spacing, False)
    acc = tf.zeros(x.shape)
    
    for i in range(spacing * spacing):
        masked, mask = masker(x, i)
        preds = model(masked)
    
        acc = acc + tf.reshape(preds, x.shape) * tf.reshape(mask, x.shape)
    return acc

class Masker:
    def __init__(self, spacing=3, interpolate=True):
        self.spacing = spacing
        self.interpolate = interpolate
        
        kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
        kernel /= kernel.sum()
        self.interpolate_kernel = kernel[:, :, np.newaxis, np.newaxis]
    
    def __call__(self, x, i=0):
        phase_x = i % self.spacing
        phase_y = (i // self.spacing) % self.spacing
    
        mask = self._create_grid(x.shape[:3], phase_x, phase_y)
        mask_inv = np.ones(mask.shape) - mask
    
        if self.interpolate:
            masked = self._apply_interpolate_mask(x, mask, mask_inv)
        else:
            masked = tf.cast(x, mask_inv.dtype) * tf.reshape(mask_inv,x.shape)
    
        return masked, mask
    
    def _create_grid(self, shape, phase_x, phase_y):
        grid = np.zeros((shape[1],shape[2]))
        for i in range(shape[1]):
            for j in range(shape[2]):
                if (i % self.spacing == phase_x and j % self.spacing == phase_y):
                    grid[i, j] = 1
        return tf.constant(np.ones([shape[0], 1, 1]) * grid, dtype=tf.float32)
    
    def _apply_interpolate_mask(self, x, mask, mask_inv):
        x64 = tf.cast(x, tf.float64)
        kernel = tf.constant(self.interpolate_kernel, dtype=tf.float64)
        filtered = tf.nn.conv2d(x64, kernel, strides=[1,1,1,1], padding='SAME')
        x_squeezed = tf.dtypes.cast(tf.squeeze(x64), dtype=mask.dtype)
        filtered_squeezed = tf.dtypes.cast(tf.squeeze(filtered), dtype=tf.float32)
        term1 = filtered_squeezed * mask
        term2 = x_squeezed  *  mask_inv
        return term1 + term2