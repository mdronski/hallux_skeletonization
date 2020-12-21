from typing import List, Tuple

from src.skeletonize import skeletonize
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


def find_lines(skeleton: np.ndarray, verbose=False) -> List[Tuple[int, int]]:
    thicker_skeleton = thicken(skeleton.copy())
    lines = find_two_lines(thicker_skeleton)

    line_1, line_2, _pts = convert_lines_to_points(lines)
    intersection_point, angle = calculate_angle(line_1, line_2, _pts)

    if verbose:
        visualize_lines(skeleton, _pts, intersection_point, angle)

    return line_1, line_2, intersection_point, angle, (_pts, intersection_point, angle)


def find_two_lines(thicker_skeleton: np.ndarray):
    rhos = np.linspace(0.5, 2.0, 20)
    thetas = np.linspace(np.pi/15, np.pi/5, 20)
    thresholds = np.linspace(80, 200, 20)

    for rho in rhos:
        for theta in thetas:
            for threshold in thresholds:
                lines = cv2.HoughLines(thicker_skeleton.astype(np.uint8),
                                       rho=rho,
                                       theta=theta,
                                       threshold=int(threshold)
                                       )

                # length = 0 if lines is None else len(lines)
                # print(f"rho: {rho}, theta{theta}, threshold{int(threshold)}, lines: {length}")
                if lines is not None:
                    if len(lines) == 2:
                        if is_angle_non_zero(lines):
                            # print(f"rho: {rho}, theta{theta}, threshold{int(threshold)}")
                            return lines
    raise Exception("Unable to find two lines")


def is_angle_non_zero(lines):
    line_1, line_2, _pts = convert_lines_to_points(lines)
    intersection_point, angle = calculate_angle(line_1, line_2, _pts)
    return angle > 20


def visualize_lines(skeleton, _pts, intersection_point, angle):
    for points in _pts:
        pt1 = points[0]
        pt2 = points[1]
        plt.plot((pt1[0], pt2[0]), (pt1[1], pt2[1]))

    if angle == 0:
        return
    ann = plt.annotate(str(angle), (intersection_point[0], intersection_point[1]), color='white')
    ann.set_fontsize(20)
    plt.scatter(intersection_point[0], intersection_point[1], c="y", marker="o")
    plt.imshow(skeleton)


def calculate_angle(line_1, line_2, lines_pts):
    intersection_point = intersection(line_1, line_2)
    angle = int(ang(lines_pts[0], lines_pts[1]))
    return intersection_point, angle


def convert_lines_to_points(lines: np.ndarray):
    lines_pts = []
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        lines_pts.append([pt1, pt2])
        # plt.plot((pt1[0], pt2[0]), (pt1[1], pt2[1]))

    l1 = line(lines_pts[0][0], lines_pts[0][1])
    l2 = line(lines_pts[1][0], lines_pts[1][1])
    return l1, l2, lines_pts


def thicken(skeleton: np.ndarray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    skeleton = cv2.dilate(skeleton.astype(np.uint8), kernel, iterations=1)
    return skeleton


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def dot(vA, vB):
    return vA[0] * vB[0] + vA[1] * vB[1]


def ang(lineA, lineB):
    vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
    vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
    dot_prod = dot(vA, vB)
    magA = dot(vA, vA) ** 0.5
    magB = dot(vB, vB) ** 0.5
    cos_ = dot_prod / magA / magB
    angle = math.acos(dot_prod / magB / magA)
    ang_deg = math.degrees(angle) % 360

    if ang_deg - 180 >= 0:
        return 360 - ang_deg
    else:

        return ang_deg
