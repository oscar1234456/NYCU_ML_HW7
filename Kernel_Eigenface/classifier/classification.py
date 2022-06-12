import numpy as np
from scipy.spatial.distance import cdist


def classification(training_z, training_labels, testing_data, testing_labels, project_matrix, k=5):
    hit = 0
    # testing_data centralized
    testing_mean_face = np.mean(testing_data, axis=0)  # (305,)
    centralized_testing_data = testing_data - testing_mean_face  # (30, 45045)

    # using project matrix to let testing data project to low dim space
    # project_matrix: (45045, kdim)
    testing_z = project_matrix.T @ centralized_testing_data.T  # (kdim, 30)

    # using knn to classify each testing data (count similarity with training data and know label)
    # training_z:(kdim, 135)
    distance_record = np.zeros(training_z.shape[1])
    # cdist(X, X, metric='sqeuclidean')
    for i in range(testing_z.shape[1]):
        for j in range(training_z.shape[1]):
            distance_record[j] = cdist(testing_z[:, i].reshape(1, -1), training_z[:, j].reshape(1, -1),
                                       metric='sqeuclidean')
        select_range = training_labels[np.argsort(distance_record)][:k]
        sort_result, sort_count = np.unique(select_range, return_counts=True)
        predict_label = sort_result[np.argmax(sort_count)]
        # print([predict_label])
        if predict_label == testing_labels[i]:

            hit += 1

    # count hit rates
    print(f"hit:{hit}")
    hit_rates = hit / testing_z.shape[1]

    return hit_rates
