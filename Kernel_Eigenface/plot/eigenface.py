import numpy as np
import matplotlib.pyplot as plt


def plot_eigenface(project_matrix, show_num=25, name="eigenface"):
    # project_matrix: (45045, k)
    if show_num > project_matrix.shape[1]:
        print("Exceed eigenvector number!")
        return
    for _ in range(show_num):
        plt.subplot(5, 5, _ + 1)
        plt.imshow(project_matrix[:, _].reshape(231, 195), cmap='gray')
        plt.axis("off")
    plt.suptitle(name)
    plt.show()
