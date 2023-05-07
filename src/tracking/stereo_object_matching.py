from skimage import io
import cv2
import scipy
import pandas as pd
from pathlib import Path
import numpy as np


def match_objects(
    results_left_df: pd.DataFrame,
    results_right_df: pd.DataFrame,
    img_left: np.ndarray,
    img_right: np.ndarray,
) -> pd.DataFrame:
    """Match objects from left and right images.

    Args:
        results_left_df (pd.DataFrame): Results from left image.
        results_right_df (pd.DataFrame): Results from right image.
        calib_file (str): Path to calibration file.

    Returns:
        pd.DataFrame: Matched objects.
    """

    # read images and convert to gray
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    nb_matches = 20
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    im_height = img_left.shape[0]
    match_matrix = np.zeros((len(results_left_df.index), len(results_right_df.index)))
    for idx_left, bbox_left in results_left_df.iterrows():
        for idx_right, bbox_right in results_right_df.iterrows():
            cy_left = bbox_left["ymin"] + bbox_left["ymax"] / 2
            cy_right = bbox_right["ymin"] + bbox_right["ymax"] / 2
            if abs(cy_left - cy_right) / im_height <= 0.1 and bbox_left[5] == bbox_right[5]:
                bbox_left_im = img_left_gray[
                    int(bbox_left["ymin"]) : int(bbox_left["ymax"]),
                    int(bbox_left["xmin"]) : int(bbox_left["xmax"]),
                ]
                bbox_right_im = img_right_gray[
                    int(bbox_right["ymin"]) : int(bbox_right["ymax"]),
                    int(bbox_right["xmin"]) : int(bbox_right["xmax"]),
                ]
                kp_left, des_left = sift.detectAndCompute(bbox_left_im, None)
                kp_right, des_right = sift.detectAndCompute(bbox_right_im, None)
                matches = bf.match(des_left, des_right)
                matches = sorted(matches, key=lambda x: x.distance)[:nb_matches]
                for m in matches:
                    match_matrix[idx_left, idx_right] += m.distance
            else:
                match_matrix[idx_left, idx_right] = 1e12

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(match_matrix)

    return row_ind, col_ind

def draw_matching_bbox(results_left_df, results_right_df, img_left, img_right, row_ind, col_ind):
    for idx_left, idx_right in zip(row_ind, col_ind):
        # draw on left image
        random_color = tuple(int(color) for color in np.random.choice(range(255),size=3))
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

