import numpy as np


def eigen_solver(target, k=-99):
    eigen_values, eigen_vectors = np.linalg.eig(target)
    keep_eigen_vectors, keep_eigen_values = eigen_vectors, eigen_values

    # let eigen value sorted from big to small
    sort_index = np.argsort(-eigen_values)
    eigen_values = eigen_values[sort_index]
    eigen_vectors = eigen_vectors[:, sort_index]

    for idx, eigen_value in enumerate(eigen_values):
        if eigen_value <= 0:  # select unuseful eigenvectors
            keep_eigen_values = eigen_values[:idx].real
            keep_eigen_vectors = eigen_vectors[:, :idx].real
            break

    if k != -99 and k > 0:
        # pick needed k
        if len(keep_eigen_values) >= k:
            print("select top k eigenVectors")
            keep_eigen_vectors = keep_eigen_vectors[:k + 1]
            keep_eigen_values = keep_eigen_values[:k + 1]
        else:
            print(f"Your k is too big. There are only "
                  f"{len(keep_eigen_values)} eigenvectors!")
            print("Using all eigenvectors")
    elif k == -99:
        print("Using all eigenvectors")
    else:
        print("Wrong k!")

    return keep_eigen_vectors, keep_eigen_values
