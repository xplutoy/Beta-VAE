import matplotlib.pyplot as plt
import numpy as np


def plot_q_z(x, y, filename):
    from sklearn.manifold import TSNE
    colors = ["#2103c8", "#0e960e", "#e40402", "#05aaa8", "#ac02ab", "#aba808", "#151515", "#94a169", "#bec9cd",
              "#6a6551"]

    plt.clf()
    fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
    if x.shape[1] != 2:
        x = TSNE().fit_transform(x)
    y = y[:, np.newaxis]
    xy = np.concatenate((x, y), axis=1)

    for l, c in zip(range(10), colors):
        ix = np.where(xy[:, 2] == l)
        ax.scatter(xy[ix, 0], xy[ix, 1], c=c, marker='o', label=l, s=10, linewidths=0)

    plt.savefig(filename)
    plt.close()
