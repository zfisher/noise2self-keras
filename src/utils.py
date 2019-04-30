import numpy as np
import matplotlib.pyplot as plt

def show_image(pixels, title=None, clip=True, output_path=None):
    show_grid([[pixels]], [title] if title else None, clip, output_path)

def show_images(images_list, titles=None, clip=True, output_path=None):
    show_grid([images_list], titles, clip, output_path)

def show_grid(grid, titles=None, clip=True, output_path=None):
    fig = plt.figure(figsize=(len(grid[0]), len(grid)))
    for i, row in enumerate(grid):
        for j, pixels in enumerate(row):
            ax = fig.add_subplot(len(grid), len(row), len(row)*i + j + 1)
            image = np.squeeze(pixels)
            if clip:
                image = np.clip(image, 0, 1)
            if titles and j == 0:
                ax.set_title(titles[i], loc='left')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image, cmap='gray_r')
    
    if titles:
        plt.subplots_adjust(hspace=0.5)
    
    if not output_path:
        plt.show()
    else:
        plt.savefig(output_path)

def show_plot(data, title=None, x_label=None, y_label=None):
    plt.plot(data)
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.show()

# adapted from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def display_progress(iteration, total, prefix = '', suffix = '', 
                     decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total: 
        print()