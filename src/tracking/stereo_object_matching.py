from skimage import io
import cv2
import scipy
import pandas as pd
from pathlib import Path
import numpy as np


def match_objects(
    results_left_df: pd.DataFrame,
    results_right_df: pd.DataFrame,
    img_left_path: Path,
    img_right_path: Path,
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
    img_left = io.imread(img_left_path)
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY)
    img_right = io.imread(img_right_path)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    nb_matches = 20
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    im_height = img_left.shape[0]
    match_matrix = np.zeros((len(results_left_df.index), len(results_right_df.index)))
    for i1, bbox1 in results_left_df.iterrows():
        for i2, bbox2 in results_right_df.iterrows():
            cy1 = bbox1["ymin"] + bbox1["ymax"] / 2
            cy2 = bbox2["ymin"] + bbox2["ymax"] / 2
            if abs(cy1 - cy2) / im_height <= 0.1 and bbox1[5] == bbox2[5]:
                bbox1_im = img_left_gray[
                    int(bbox1["ymin"]) : int(bbox1["ymax"]),
                    int(bbox1["xmin"]) : int(bbox1["xmax"]),
                ]
                bbox2_im = img_right_gray[
                    int(bbox2["ymin"]) : int(bbox2["ymax"]),
                    int(bbox2["xmin"]) : int(bbox2["xmax"]),
                ]
                kp1, des1 = sift.detectAndCompute(bbox1_im, None)
                kp2, des2 = sift.detectAndCompute(bbox2_im, None)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)[:nb_matches]
                for m in matches:
                    match_matrix[i1, i2] += m.distance
            else:
                match_matrix[i1, i2] = np.inf

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(match_matrix)

    return row_ind, col_ind
