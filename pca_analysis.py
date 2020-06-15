from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import src
from reading_data import Data


def pca_cumulative_variance(tensor_features):
    print("Input shape: {}".format(tensor_features.shape))
    _pca = PCA().fit(tensor_features)
    plt.plot(np.cumsum(_pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


def pca_scatter_plot_3d(tensor_features, _labels, str_labels):
    _pca = PCA(3)
    projected = _pca.fit_transform(tensor_features)
    print("Input shape {}".format(tensor_features.shape))
    print("Output shape {}".format(projected.shape))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c=_labels)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper left", title="Classes")
    ax.add_artist(legend1)
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.set_ylabel('component 3')
    plt.show()


def pca_scatter_plot(tensor_features, _labels):
    _pca = PCA(2)
    projected = _pca.fit_transform(tensor_features)
    print("Input shape {}".format(tensor_features.shape))
    print("Output shape {}".format(projected.shape))

    plt.scatter(projected[:, 0], projected[:, 1],
                c=labels, edgecolor='none', alpha=0.9)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dt = Data()
    tensor_images, labels = dt.get_tensor_images("train")
    pca_cumulative_variance(tensor_images)
