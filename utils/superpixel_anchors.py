import numpy as np
from skimage.measure import regionprops
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import minmax_scale


def create_association_mat(superpixel_labels):
    labels = np.unique(superpixel_labels)
    # print(labels)
    n_labels = labels.shape[0]
    print('num superpixel: ', n_labels)
    n_pixels = superpixel_labels.shape[0] * superpixel_labels.shape[1]
    association_mat = np.zeros((n_pixels, n_labels))
    superpixel_labels_ = superpixel_labels.reshape(-1)
    for i, label in enumerate(labels):
        association_mat[np.where(label == superpixel_labels_), i] = 1
    return association_mat


def create_spixel_graph(source_img, superpixel_labels):
    s = source_img.reshape((-1, source_img.shape[-1]))
    a = create_association_mat(superpixel_labels)
    # t = superpixel_labels.reshape(-1)
    mean_fea = np.matmul(a.T, s)
    regions = regionprops(superpixel_labels + 1)
    n_labels = np.unique(superpixel_labels).shape[0]
    center_indx = np.zeros((n_labels, 2))
    for i, props in enumerate(regions):
        center_indx[i, :] = props.centroid  # centroid coordinates
    ss_fea = np.concatenate((mean_fea, center_indx), axis=1)
    ss_fea = minmax_scale(ss_fea)
    adj = kneighbors_graph(ss_fea, n_neighbors=50, mode='distance', include_self=False).toarray()

    # # auto calculate gamma in Gaussian kernel
    X_var = ss_fea.var()
    gamma = 1.0 / (ss_fea.shape[1] * X_var) if X_var != 0 else 1.0
    adj[np.where(adj != 0)] = np.exp(-np.power(adj[np.where(adj != 0)], 2) * gamma)

    # adj = euclidean_dist(ss_fea, ss_fea).numpy()
    # adj = np.exp(-np.power(adj, 2) * gamma)
    np.fill_diagonal(adj, 0)

    # show_graph(adj, center_indx)
    return adj, center_indx


def generate_anchors(x_list, superpixel_labels):
    """
    :param x_list: [[n_sample, n_dim], ...]
    :param superpixel_labels:
    :return:
    """
    anchors = []
    for x_i, seg_i in zip(x_list, superpixel_labels):
        association = create_association_mat(seg_i)
        association /= association.sum(axis=0)
        mean_fea = np.matmul(association.T, x_i)

        # # graph filtering
        # adj = kneighbors_graph(mean_fea, n_neighbors=3, mode='connectivity', include_self=False).toarray()
        # adj = (adj + adj.T)/2 + np.eye(adj.shape[0])
        # D = np.diag(np.power(np.sum(adj, axis=1), -0.5))
        # norm_adj = np.matmul(np.matmul(D, adj), D)
        # L = np.eye(adj.shape[0]) - norm_adj
        # mean_fea = np.matmul(np.linalg.inv(np.eye(adj.shape[0]) + 0.5 * L), mean_fea)

        anchors.append(mean_fea)
    return anchors