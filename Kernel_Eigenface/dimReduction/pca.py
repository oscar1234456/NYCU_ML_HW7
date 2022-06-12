import numpy as np

from Kernel_Eigenface.eigenProblem.solver import eigen_solver
from Kernel_Eigenface.kernel.get_kernel import get_kernel


def pca(training_data, kernel_option="no"):
    # count mean (mean face)
    mean_face = np.mean(training_data, axis=0)  # (135,)

    # let data points centralized -> zero mean
    centralized_training_data = training_data - mean_face  # (135, 45045)

    # get kernel <kernel pca> or covariance matrix <pca>
    # notice: assumed data are centered already
    S = get_kernel(centralized_training_data, kernel_option=kernel_option)

    # find k principal components (solve eigenvalue/eigenvector problem) (135, k)
    keep_eigen_vectors, keep_eigen_values = eigen_solver(S, k=-99)

    # compose the project matrix (42045, 135)@(135, k) = (42045, k)
    project_matrix = centralized_training_data.T @ keep_eigen_vectors

    # normalize principal components

    # print eigenface based on project matrix
    # count z (low dim representation) (k, 42045) @ (42045, 135) = (k, 135)
    # reconstruction z (42045, k) @ (k, 135) = (42045, 135)
    # classification
    pass


def get_project_matrix(centralized_data, eigen_vectors, normalization=True):
    # centralized data (135, 45045)
    # eigen_vectors (135, k)
    project_matrix = centralized_data.T @ eigen_vectors

    if normalization:
        project_matrix_norm = np.linalg.norm(project_matrix, axis=0)
        project_matrix = project_matrix / project_matrix_norm
    return project_matrix
