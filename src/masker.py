import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def infer(x, model, spacing=3):
    masker = Masker(False, spacing)
    result = tf.zeros(x.shape)
    
    for i in range(spacing * spacing):
        masked, mask = masker(x, i)
        masked = tf.reshape(masked,(-1, *x.shape[1:]))
        predictions = model(tf.cast(masked, tf.float32))
    
        result += tf.reshape(predictions, x.shape) * tf.reshape(mask, x.shape)
    return result

class Masker:
    def __init__(self, interpolate=True, spacing=3, radius=1, smooth=True):
        self.spacing = spacing
        self.interpolate = interpolate
        
        # build a disk-shaped matrix to convolve with.
        x, y = np.ogrid[-radius:(radius+1), -radius:(radius+1)]
        within_outer_radius = (x*x + y*y <= (radius+1)*(radius+1))
        within_inner_radius = (x*x + y*y <= radius*radius)
        kernel = np.zeros((2*radius + 1, 2*radius + 1))
        if smooth:
            kernel[within_outer_radius] = 0.5 # fill in outer disk 
        kernel[within_inner_radius] = 1.0 # fill in inner disk 
        kernel[radius,radius] = 0.0 # subtract center point
        kernel /= kernel.sum()
        self.interpolate_kernel = kernel[:, :, np.newaxis, np.newaxis]
    
    def __call__(self, x, i=0, shape=None):
        phase_x = i % self.spacing
        phase_y = (i // self.spacing) % self.spacing
    
        mask = self._create_grid(x.shape[:3], phase_x, phase_y)
        mask_inv = np.ones(mask.shape) - mask
    
        if self.interpolate:
            masked = self._apply_interpolate_mask(x, mask, mask_inv)
        else:
            masked = x * tf.cast(tf.reshape(mask_inv, x.shape), x.dtype)
        
        if shape:
            masked = tf.reshape(masked, shape)
            mask = tf.reshape(mask, shape)
    
        return masked, mask
    
    def _create_grid(self, shape, phase_x, phase_y):
        grid = np.zeros((shape[1],shape[2]))
        for i in range(shape[1]):
            for j in range(shape[2]):
                if (i % self.spacing == phase_x and j % self.spacing == phase_y):
                    grid[i, j] = 1
        grid = np.ones([shape[0], 1, 1]) * grid
        return tf.constant(grid, dtype=tf.float32)
    
    def _apply_interpolate_mask(self, x, mask, mask_inv):
        x64 = tf.cast(x, tf.float64)
        kernel = tf.constant(self.interpolate_kernel, dtype=tf.float64)
        filtered = tf.nn.conv2d(x64, kernel, strides=[1,1,1,1], padding='SAME')
        x_squeezed = tf.dtypes.cast(tf.squeeze(x64), dtype=mask.dtype)
        filtered_squeezed = tf.dtypes.cast(tf.squeeze(filtered), dtype=tf.float32)
        return (filtered_squeezed * mask) + (x_squeezed * mask_inv)