import random

import numpy as np
import matplotlib.pyplot as plt


def plot_reconstruction(data, reconstruction_data, show_num=10, name="PCA Reconstruction"):
    # data: (135, 45045)
    # reconstruction: (45045, 135)
    select_index = random.sample(range(data.shape[0]), show_num)
    for idx, sel_idx in enumerate(select_index):
        plt.subplot(show_num, show_num, idx + 1)
        plt.imshow(data[sel_idx, :].reshape(231, 195), cmap='gray')
        plt.axis("off")
        plt.subplot(show_num, show_num, idx + (show_num) + 1)
        plt.imshow(reconstruction_data[:, sel_idx].reshape(231, 195), cmap='gray')
        plt.axis("off")
    plt.suptitle(name)
    plt.show()
