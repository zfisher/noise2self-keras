import numpy as np
import matplotlib.pyplot as plt

def show_image(pixels):
    image = (np.reshape(pixels, (28,28))*255).astype(np.uint8)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='gray_r')
    plt.show()

def show_images(images_list):
    fig = plt.figure(figsize=(1,len(images_list)))
    for i, pixels in enumerate(images_list):
        fig.add_subplot(1, len(images_list), i+1)
        image = (np.reshape(pixels, (28,28))*255).astype(np.uint8)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap='gray_r')
    plt.show()

def show_grid(grid):
    fig = plt.figure(figsize=(len(grid),len(grid[0])))
    for i, row in enumerate(grid):
        for j, pixels in enumerate(row):
            fig.add_subplot(len(grid), len(row), len(row)*i + j + 1)
            image = (np.reshape(pixels, (28,28))*255).astype(np.uint8)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image, cmap='gray_r')
    plt.show()

def show_plot(data, *args, **kwargs):
    plt.plot(data, *args, **kwargs)
    plt.show()

# source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def display_progress (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total: 
        print()