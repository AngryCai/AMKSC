from Preprocessing import Processor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


def order_sample_for_graph_show(x, y):
    x_new = np.zeros(x.shape, dtype=np.float32)
    y_new = np.zeros(y.shape, dtype=np.int8)
    start = 0
    for i in np.unique(y):
        idx = np.nonzero(y == i)
        stop = start + idx[0].shape[0]
        x_new[start:stop] = x[idx]
        y_new[start:stop] = y[idx]
        start = stop
    return x_new, y_new


def load_multimodal_data(gt_path, *src_path, patch_size=(7, 7), is_labeled=True):
    p = Processor()
    n_modality = len(src_path)
    modality_list = []
    in_channels = []
    for i in range(n_modality):
        img, gt = p.prepare_data(src_path[i], gt_path)
        # img, gt = img[:, :100, :], gt[:, :100]
        x_patches, y_ = p.get_HSI_patches_rw(img, gt, (patch_size[0], patch_size[1]), is_indix=False,
                                             is_labeled=is_labeled)
        # x_patches, y_ = order_sample_for_graph_show(x_patches, y_)
        n_samples, n_row, n_col, n_channel = x_patches.shape

        scaler = StandardScaler()
        batch_size = 5000
        # # using incremental / batch for very large data
        for start_id in range(0, x_patches.shape[0], batch_size):
            n_batch = x_patches[start_id: start_id + batch_size].shape[0]
            scaler.partial_fit(x_patches[start_id: start_id + batch_size].reshape(n_batch, -1))
        for start_id in range(0, x_patches.shape[0], batch_size):
            shape = x_patches[start_id: start_id + batch_size].shape
            x_temp = x_patches[start_id: start_id + batch_size].reshape(shape[0], -1)
            x_patches[start_id: start_id + batch_size] = scaler.transform(x_temp).reshape(shape)

        # x_patches = scale(x_patches.reshape((n_samples, -1)))  # .reshape((n_samples, n_row, n_col, -1))
        x_patches = x_patches.reshape((n_samples, -1))
        # x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
        modality_list.append(x_patches)
        in_channels.append(n_channel)
    y = p.standardize_label(y_)
    return modality_list, y, gt



