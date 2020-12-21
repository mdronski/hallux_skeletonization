from src.angle_find import find_lines
from src.data_load import load_single, load_all
from src.skeletonize import skeletonize
import numpy as np
import matplotlib.pyplot as plt


def run():
    for img in load_all():
        run_for_single(img)


def run_for_single(img: np.ndarray):
    skeleton_raw, skeleton_processed = skeletonize(img)
    _, _, _, _, line_data = find_lines(skeleton_processed)

    visualize_all(img, skeleton_raw, skeleton_processed, line_data)


def visualize_all(img, skeleton_raw, skeleton_processed, line_data):
    fig, axes = plt.subplots(nrows=1,
                             ncols=4,
                             figsize=(20, 10),
                             sharex=True,
                             sharey=True
                             )
    ax = axes.ravel()

    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=15)

    ax[1].imshow(skeleton_raw)
    ax[1].axis('off')
    ax[1].set_title('skeleton_raw', fontsize=15)

    ax[2].imshow(skeleton_processed)
    ax[2].axis('off')
    ax[2].set_title('skeleton_processed', fontsize=15)

    ax[3].axis('off')
    (_pts, intersection_point, angle) = line_data
    for points in _pts:
        pt1 = points[0]
        pt2 = points[1]
        ax[3].plot((pt1[0], pt2[0]), (pt1[1], pt2[1]))

    ann = ax[3].annotate(str(angle), (intersection_point[0], intersection_point[1]), color='white')
    ann.set_fontsize(15)
    ax[3].scatter(intersection_point[0], intersection_point[1], c="y", marker="o")
    ax[3].set_title('found lines', fontsize=15)
    ax[3].imshow(skeleton_processed)

    plt.show()