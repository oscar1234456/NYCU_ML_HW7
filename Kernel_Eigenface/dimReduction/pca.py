import numpy as np
from matplotlib import pyplot as plt

from Kernel_Eigenface.classifier.classification import classification
from Kernel_Eigenface.eigenProblem.solver import eigen_solver
from Kernel_Eigenface.kernel.get_kernel import get_kernel
from Kernel_Eigenface.plot.eigenface import plot_eigenface
from Kernel_Eigenface.plot.reconstruction import plot_reconstruction


def pca(training_data, kernel_option="no",
        k=-99,
        normalization=True,
        show_eigenface_num=25,
        kernel_name="no"):
    # count mean (mean face)
    mean_face = np.mean(training_data, axis=0)  # (135,)

    # let data points centralized -> zero mean
    centralized_training_data = training_data - mean_face  # (135, 45045)

    # get kernel <kernel pca> or covariance matrix <pca>
    # notice: assumed data are centered already
    S = get_kernel(centralized_training_data, kernel_option=kernel_option)

    # find k principal components (solve eigenvalue/eigenvector problem) (135, k)
    keep_eigen_vectors, keep_eigen_values = eigen_solver(S, k=k)

    # compose the project matrix (42045, 135)@(135, k) = (42045, k)
    project_matrix = get_project_matrix(centralized_training_data,
                                        keep_eigen_vectors,
                                        normalization=normalization)

    # print eigenface based on project matrix
    plot_eigenface(project_matrix, show_num=show_eigenface_num, name="EigenFace" + f" (kernel:{kernel_name})")

    # count z (low dim representation) (k, 42045) @ (42045, 135) = (k, 135)
    z = get_pca_z(project_matrix, centralized_training_data)

    # reconstruction z (42045, k) @ (k, 135) = (42045, 135)
    training_data_reconstruction = reconstruction(project_matrix, z,
                                                  mean_face, add_mean=True)

    # plot reconstruction
    plot_reconstruction(training_data, training_data_reconstruction, show_num=10,
                        name="PCA Reconstruction" + f" (kernel:{kernel_name})")

    return z, project_matrix


def get_project_matrix(centralized_data, eigen_vectors, normalization=True):
    # centralized data (135, 45045)
    # eigen_vectors (135, k)
    project_matrix = centralized_data.T @ eigen_vectors

    if normalization:
        # normalize principal components
        project_matrix_norm = np.linalg.norm(project_matrix, axis=0)
        project_matrix = project_matrix / project_matrix_norm
    return project_matrix


def get_pca_z(project_matrix, data):
    return project_matrix.T @ data.T


def reconstruction(project_matrix, z, mean_face, add_mean=True):
    # mean_face: (135,)
    return project_matrix @ z + mean_face.reshape(-1, 1) if add_mean else project_matrix @ z
