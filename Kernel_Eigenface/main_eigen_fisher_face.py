import numpy as np

from Kernel_Eigenface.classifier.classification import classification
from Kernel_Eigenface.dataloader.imageloader import load_image
from Kernel_Eigenface.dimReduction.lda import lda
from Kernel_Eigenface.dimReduction.pca import pca

# Config
only_pca = True
kernel_options = ["no", "linear", "polynomial", "RBF"]
# kernel_options = ["no"]

# load image
# training_data: (135, 42045), training_label: (135,)
# testing_data: (30, 42045), testing_label: (30,)
training_data, training_labels = load_image("./Yale_Face_Database/Training")
testing_data, testing_labels = load_image("./Yale_Face_Database/Testing")

for kernel_option in kernel_options:
    # call PCA -> return the data points representation
    #             with low dim and transform matrix
    print("==============")
    # print(f"kernel:{kernel_option}")
    pca_z, pca_project_matrix = pca(training_data, kernel_option=kernel_option,
                                    k=-99,
                                    normalization=True,
                                    show_eigenface_num=25, kernel_name=kernel_option)

    # classification PCA
    hit_rate_pca = classification(pca_z, training_labels, testing_data, testing_labels, pca_project_matrix, k=5)
    print(f"PCA Classification hit rates:{hit_rate_pca}")

    # call LDA -> put representations in low dim and transform matrix
    lda_z, lda_project_matrix = lda(pca_project_matrix, pca_z, training_data,
                                    k=-99, show_fisherface_num=25,
                                    kernel_name=kernel_option)
    # classification LDA
    hit_rate_lda = classification(lda_z, training_labels, testing_data, testing_labels, lda_project_matrix, k=5)
    print(f"LDA Classification hit rates:{hit_rate_lda}")

    print("==============")
