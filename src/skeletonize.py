from typing import Tuple

import cv2
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd


def skeletonize(img: np.ndarray, verbose=False) -> np.ndarray:
    skeleton = get_best_skeleton(img)
    skeleton_raw = skeleton.copy()
    remove_branches(skeleton)

    if verbose:
        visualize_skeleton(img, skeleton, skeleton_raw)

    return skeleton_raw, skeleton


def get_best_skeleton(img):
    s1 = morphology.skeletonize(img)
    s2 = morphology.medial_axis(img)
    s3 = morphology.thin(img)

    values = np.asarray([s1, s2])
    sums = np.asarray([s1.sum(), s2.sum()])
    return values[np.argmin(sums)]


def remove_branches(skeleton: np.ndarray) -> None:
    branchpoints, endpoints = find_branch_end_points(skeleton)
    x, y = np.where((endpoints == 255))
    endpoint_list = list(zip(x, y))
    for endpoint in endpoint_list:
        remove_branch(skeleton, branchpoints, endpoint)


def remove_branch(skeleton: np.ndarray, branchpoints: np.ndarray, endpoint: Tuple[int, int]) -> None:
    endpoint_img = np.zeros_like(skeleton)
    endpoint_img[endpoint[0], endpoint[1]] = 1
    dist = nd.distance_transform_edt(endpoint_img == 0)
    dist_to_closest_branchpoint = np.min(dist[branchpoints == 255])
    if (dist_to_closest_branchpoint < 100):
        skeleton[dist < dist_to_closest_branchpoint] = 0


def find_branch_end_points(skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    skel = skeleton.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)

    branch_points = np.zeros_like(skel)
    end_points = np.zeros_like(skel)
    branch_points[np.where(filtered >= 13)] = 255
    end_points[np.where(filtered == 11)] = 255
    return branch_points, end_points


def visualize_skeleton(img: np.ndarray, skel: np.ndarray, skel_raw: np.ndarray) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=3,
                             sharex=True, sharey=True
                             )
    ax = axes.ravel()

    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=10)

    ax[1].imshow(skel_raw)
    ax[1].axis('off')
    ax[1].set_title('skeleton_raw', fontsize=10)

    ax[2].imshow(skel)
    ax[2].axis('off')
    ax[2].set_title('skeleton_processed', fontsize=10)

    fig.tight_layout()
    plt.show()
