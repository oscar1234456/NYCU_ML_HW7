#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.
import os

import numpy as np
import pylab
from PIL import Image
from scipy.spatial.distance import cdist


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    # D is same as a = cdist(X, X, metric='sqeuclidean')
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, labels=None, symmetric=False):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real  # pca dim 784 reduction to 50
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.  # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        if symmetric:
            # ssne
            num = np.exp(-1 * np.add(np.add(num, sum_Y).T, sum_Y))
        else:
            # tsne
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))  # inverse= 倒數
        # np.add(np.add(num, sum_Y).T, sum_Y) same as cdist(Y,Y, "sqeuclidean")
        num[range(n), range(n)] = 0.  # 對角線設為0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        if symmetric:
            # ssne
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        else:
            # tsne
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            # save pic
            if symmetric:
                # symmetric sne
                show_pic(Y, labels, [-10, 10], iter + 1, save=True, target_dir=f"./ssne/ssne_{int(perplexity)}")
            else:
                show_pic(Y, labels, [-120, 120], iter + 1, save=True, target_dir=f"./tsne/tsne_{int(perplexity)}")

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    show_similarity_dist(P, Q, symmetric, perplexity, save=True)
    # Return solution
    return Y


def show_pic(Y, labels, lim, iter, save=True, target_dir=None):
    pylab.clf()
    pylab.xlim(lim)
    pylab.ylim(lim)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.title(target_dir[2:6] + f" (perplexity:{target_dir[12:14]} / Iter:{iter})")
    if save:
        pylab.savefig(target_dir + f"/{iter}.png")
    # pylab.show()


def convert_GIF(path):
    images = list()
    # path: ./tsne/tsne_10
    pic_filenames = os.listdir(path)
    pic_filenames.sort(key=lambda x: int(x[:-4]))
    for pic_filename in pic_filenames:
        images.append(Image.open(path + "/" + pic_filename))
    if len(images) == 0:
        print("There are not any pics in this folder!")
        return
    images[0].save(f'{path[:7]}/{path[7:]}.gif',
                   save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
    print("convert GIF done!")


def show_similarity_dist(P, Q, symmetric, perplexity, save=True):
    method = "ssne" if symmetric else "tsne"
    pylab.subplot(2, 1, 1)
    pylab.title(f"{method} High-dimensional space (P)", fontsize=7)
    pylab.hist(P.flatten(), bins=35, log=True)  # P:(2500,2500)
    pylab.subplot(2, 1, 2)
    pylab.title(f"{method} Low-dimensional space (Q)", fontsize=7)
    pylab.hist(Q.flatten(), bins=35, log=True)  # Q:(2500,2500)
    if save:
        pylab.savefig(f"./{method}/similarity_dist_{int(perplexity)}.png")
    # pylab.show()


if __name__ == "__main__":
    # try:
    #     os.mkdir("./tsne")
    #     os.mkdir("./ssne")
    #     for perplexity in ["10", "20", "30", "40", "50"]:
    #         os.mkdir("./tsne_" + perplexity)
    #         os.mkdir("./ssne_" + perplexity)
    # except OSError:
    #     print("dir already exist")
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    for symmetric in [True, False]:
        word = "ssne" if symmetric else "tsne"
        for perplexity in [10.0, 20.0, 30.0, 40.0, 50.0]:
            Y = tsne(X, 2, 50, perplexity, labels, symmetric=symmetric)
            convert_GIF(f"./{word}/{word}_{int(perplexity)}")
