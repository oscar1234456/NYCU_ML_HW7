import numpy as np

from Kernel_Eigenface.dataloader.imageloader import load_image

# Config
from Kernel_Eigenface.dimReduction.pca import pca

only_pca = True
kernel_options = ["no", "linear", "polynomial", "RBF"]

# load image
# training_data: (135, 42045), training_label: (135,)
training_data, training_label = load_image("./Yale_Face_Database/Training")

for kernel_option in kernel_options:
    # call PCA -> return the data points representation
    #             with low dim and transform matrix
    pca()
    # call LDA -> put representations in low dim and transform matrix
    pass
