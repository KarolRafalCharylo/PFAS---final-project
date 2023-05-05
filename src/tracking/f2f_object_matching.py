from skimage import io
import cv2
import scipy
import pandas as pd
from pathlib import Path
import numpy as np


def get_bbox_points(
    cx: float, cy: float, w: float, h: float
) -> tuple[float, float, float, float]:
    """Get bounding box points.

    Args:
        cx (float): Center x coordinate.
        cy (float): Center y coordinate.
        w (float): Width.
        h (float): Height.

    Returns:
        tuple: Bounding box points.

    """

    # get bounding box points
    xmin = cx - w / 2
    xmax = cx + w / 2
    ymin = cy - h / 2
    ymax = cy + h / 2

    return xmin, xmax, ymin, ymax


def match_objects_f2f(
    latest_frame_df: pd.DataFrame,
    prev_frame_df: pd.DataFrame,
    img_latest: np.ndarray,
    img_prev: np.ndarray,
) -> pd.DataFrame:
    # generate docstring
    """Match objects from latest and previous frames.

    Args:
        latest_frame_df (pd.DataFrame): Results from latest frame.
        prev_frame_df (pd.DataFrame): Results from previous frame.
        img_latest (np.ndarray): Latest frame.
        img_prev (np.ndarray): Previous frame.

    Returns:
        pd.DataFrame: Matched objects.

    """

    # latest_frame_df.reset_index(drop=True, inplace=True)
    # prev_frame_df.reset_index(drop=True, inplace=True)

    # read images and convert to gray
    img_latest_gray = cv2.cvtColor(img_latest, cv2.COLOR_RGB2GRAY)
    img_prev_gray = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)

    nb_matches = 20
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    im_height = img_latest.shape[0]
    match_matrix = np.zeros((len(latest_frame_df.index), len(prev_frame_df.index)))
    for idx_latest, (_, bbox_latest) in enumerate(latest_frame_df.iterrows()):
        for idx_prev, (_, bbox_prev) in enumerate(prev_frame_df.iterrows()):
            cy_latest = bbox_latest["bbox_cy"]
            cy_prev = bbox_prev["bbox_cy"]

            if (
                abs(cy_latest - cy_prev) / im_height <= 0.1
                and bbox_latest["object_type"] == bbox_prev["object_type"]
            ):
                xmin_latest, xmax_latest, ymin_latest, ymax_latest = get_bbox_points(
                    bbox_latest["bbox_cx"],
                    bbox_latest["bbox_cy"],
                    bbox_latest["bbox_w"],
                    bbox_latest["bbox_h"],
                )
                xmin_prev, xmax_prev, ymin_prev, ymax_prev = get_bbox_points(
                    bbox_prev["bbox_cx"],
                    bbox_prev["bbox_cy"],
                    bbox_prev["bbox_w"],
                    bbox_prev["bbox_h"],
                )

                bbox_latest_im = img_latest_gray[
                    int(ymin_latest) : int(ymax_latest),
                    int(xmin_latest) : int(xmax_latest),
                ]
                bbox_prev_im = img_prev_gray[
                    int(ymin_prev) : int(ymax_prev), int(xmin_prev) : int(xmax_prev)
                ]

                kp_latest, des_latest = sift.detectAndCompute(bbox_latest_im, None)
                kp_prev, des_prev = sift.detectAndCompute(bbox_prev_im, None)
                matches = bf.match(des_latest, des_prev)
                matches = sorted(matches, key=lambda x: x.distance)[:nb_matches]
                for m in matches:
                    match_matrix[idx_latest, idx_prev] += m.distance
            else:
                match_matrix[idx_latest, idx_prev] = 1e12

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(match_matrix)

    # get indices of matched objects if they are of the same type
    indices = [
        (latest_frame_df.iloc[[ri]].index, prev_frame_df.iloc[[ci]].index)
        for ri, ci in zip(row_ind, col_ind)
        if match_matrix[ri, ci] < 1e10
    ]

    df_latest_idx, df_prev_idx = zip(*indices)
    # df_prev_idx = [prev_frame_df.iloc[[ci]].index for ci in col_ind]

    return df_latest_idx, df_prev_idx


def draw_matching_bbox(
    results_left_df, results_right_df, img_left, img_right, row_ind, col_ind
):
    for idx_left, idx_right in zip(row_ind, col_ind):
        # draw on left image
        random_color = tuple(
            int(color) for color in np.random.choice(range(255), size=3)
        )
        box_left = results_left_df.iloc[idx_left]
        x_min = int(box_left["xmin"])
        y_min = int(box_left["ymin"])
        x_max = int(box_left["xmax"])
        y_max = int(box_left["ymax"])
        cv2.rectangle(img_left, (x_min, y_min), (x_max, y_max), random_color, 2)

        # draw on right image
        box_right = results_right_df.iloc[idx_right]
        x_min = int(box_right["xmin"])
        y_min = int(box_right["ymin"])
        x_max = int(box_right["xmax"])
        y_max = int(box_right["ymax"])
        cv2.rectangle(img_right, (x_min, y_min), (x_max, y_max), random_color, 2)

    merged = np.concatenate((img_left, img_right), axis=0)
    io.imsave("matching_bbox.png", merged)
