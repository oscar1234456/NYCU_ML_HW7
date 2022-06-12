import numpy as np

from Kernel_Eigenface.dimReduction.pca import get_project_matrix
from Kernel_Eigenface.eigenProblem.solver import eigen_solver
from Kernel_Eigenface.plot.eigenface import plot_eigenface
from Kernel_Eigenface.plot.reconstruction import plot_reconstruction


def lda(pca_project_matrix, pca_z, training_data, k=-99, show_fisherface_num=25, kernel_name="no"):
    # pca_project_matrix: (45045, kdim)
    # pca_z: (kdim, 135)

    # center training data
    mean_face = np.mean(training_data, axis=0)
    # let data points centralized -> zero mean
    centralized_training_data = training_data - mean_face  # (135, 45045)

    # get within-class scatter S_W (kdim * kdim)
    S_W = get_within_class_scatter(pca_z)

    # get between-class scatter S_B (kdim * kdim)
    S_B = get_between_class_scatter(pca_z)

    # get final S = inv(S_W) @ S_B
    S = np.linalg.inv(S_W) @ S_B

    # solve eigenvector/values problem with S
    keep_eigen_vectors, keep_eigen_values = eigen_solver(S, k=k)

    # compose the project matrix
    project_matrix = get_project_matrix(pca_project_matrix,
                                        keep_eigen_vectors,
                                        normalization=False)
    # print eigenface based on project matrix
    plot_eigenface(project_matrix, show_num=show_fisherface_num, name="FisherFace" + f" (kernel:{kernel_name})")

    # project training data into low dim space z
    lda_z = get_lda_z(project_matrix, centralized_training_data)

    # reconstruct z representation to original dim space
    training_data_reconstruction = reconstruction(project_matrix, lda_z, mean_face, add_mean=True)

    # plot reconstruction
    plot_reconstruction(training_data, training_data_reconstruction, show_num=10,
                        name="LDA Reconstruction" + f" (kernel:{kernel_name})")

    return lda_z, project_matrix


def get_within_class_scatter(pca_z):
    # total 15 subjects(classes)
    # each class has 9 pictures
    # follow the lecture notes notation
    # make sure the input data is sorted well (from first subjects to final subjects)
    # pca_z: (kdim, 135)
    k = 15
    C_j = 9
    S_W = np.zeros((pca_z.shape[0], pca_z.shape[0]))
    for j in range(k):
        S_W += C_j * np.cov(pca_z[:, j * C_j:j * C_j + C_j], bias=True)

    return S_W


def get_between_class_scatter(pca_z):
    # total 15 subjects(classes)
    # each class has 9 pictures
    # follow the lecture notes notation
    # make sure the input data is sorted well (from first subjects to final subjects)
    # pca_z: (kdim, 135)
    mean_tile = np.mean(pca_z, axis=1)
    k = 15
    C_j = 9
    S_B = np.zeros((pca_z.shape[0], pca_z.shape[0]))

    for j in range(k):
        within_class_mean = np.mean(pca_z[:, j * C_j:j * C_j + C_j], axis=1).T
        S_B += C_j * (within_class_mean - mean_tile) @ (within_class_mean - mean_tile).T

    return S_B


def get_lda_z(project_matrix, data):
    # project_matrix: (45045, new_k_dim)
    # data (135, 45045)
    return project_matrix.T @ data.T


def reconstruction(project_matrix, z, mean_face, add_mean=True):
    # mean_face: (135,)
    return project_matrix @ z + mean_face.reshape(-1, 1) if add_mean else project_matrix @ z


def get_project_matrix(pca_project_matrix, eigen_vectors, normalization=True):
    # pca_project_matrixa (45045, kdim)
    # eigen_vectors (kdim, new_kdim)
    project_matrix = pca_project_matrix @ eigen_vectors  # (45045, new_kdim)

    if normalization:
        # normalize principal components
        project_matrix_norm = np.linalg.norm(project_matrix, axis=0)
        project_matrix = project_matrix / project_matrix_norm
    return project_matrix
