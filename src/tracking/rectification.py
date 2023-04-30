import numpy as np
import cv2
import os
from pathlib import Path
import click
import json

nb_matches = 200
ransac_reproj_threshold = 1
ransac_confidence = 0.95

@click.command()
@click.option(
    '--calib_image_1_path',
    default='../../data/raw/final_project/calib/image_02/data/0000000000.png',
    prompt='Calibration image 1 path: ',
    help='Choose the path to the first image for the rectification calibration!',
    type=click.Path()
)
@click.option(
    '--calib_image_2_path',
    default='../../data/raw/final_project/calib/image_03/data/0000000000.png',
    prompt='Calibration image 2 path: ',
    help='Choose the path to the second image for the rectification calibration!',
    type=click.Path()
)
@click.option(
    '--calib_file_path',
    default='../models/rectification_calib.json',
    prompt='Calibration output json file: ',
    help='Chose the path and name of the json file where to put the calibration data from the rectification calibration!',
    type=click.Path()
)

def save_parameters_to_json(H1, H2, calib_file_path) -> None:
    data_dict = {'H1': H1.tolist(), 'H2': H2.tolist()}
    with open(calib_file_path, 'w') as output_file:
        json.dump(data_dict, output_file)

def load_parameters_from_json(calib_file_path):
    with open(calib_file_path, 'r') as input_file:
        data_dict = json.load(input_file)
    return np.asarray(data_dict['H1']), np.asarray(data_dict['H2'])

def calibrate(calib_image_1_path: Path, calib_image_2_path: Path, calib_file_path: Path) -> None:
    assert str(calib_file_path)[-5:] == '.json', 'Calibration output path has to be a json file.'
    if not os.path.isdir(os.path.split(calib_file_path)[0]):
        os.mkdir(os.path.split(calib_file_path)[0], parents=True)
    assert os.path.isfile(calib_image_1_path), 'Calibration input image 1 does not exist.'
    assert os.path.isfile(calib_image_2_path), 'Calibration input image 2 does not exist.'
    image1 = cv2.imread(str(calib_image_1_path))
    image2 = cv2.imread(str(calib_image_2_path))
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1_gray, None)
    kp2, des2 = sift.detectAndCompute(image2_gray, None)
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    pts1 = []
    pts2 = []
    for m in matches[:nb_matches]:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=ransac_reproj_threshold, confidence=ransac_confidence)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    ret, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=image1_gray.shape)
    assert ret, 'Rectification homographies could not be calculated.'
    save_parameters_to_json(H1, H2, calib_file_path)
    print(str(calib_file_path), 'was created.')

def rectify(image1: np.ndarray, image2: np.ndarray, calib_file_path: Path):
    assert str(calib_file_path)[-5:] == '.json', 'Calibration file has to be a json file.'
    assert os.path.isfile(calib_file_path), 'Calibration file does not exist.'
    H1, H2 = load_parameters_from_json(calib_file_path)
    image1_rect = cv2.warpPerspective(image1, H1, (image1_rect.shape[1], image1_rect.shape[0]))
    image2_rect = cv2.warpPerspective(image2, H2, (image2_rect.shape[1], image2_rect.shape[0]))
    return image1_rect, image2_rect

if __name__ == '__main__':
    calibrate()