import numpy as np
import cv2
import os
from pathlib import Path
import click
import json

@click.command()
@click.option(
    '--calib_image_path',
    default='../../data/raw/final_project/calib/image_02/data/0000000000.png',
    prompt='Calibration image path: ',
    help='Choose the path to the calibration image (with checkerboards) for the undistortion calibration!',
    type=click.Path()
)
@click.option(
    '--calib_file_path',
    default='../models/undistortion_calib_image_02.json',
    prompt='Calibration output json file: ',
    help='Chose the path and name of the json file where to put the calibration data from the undistortion calibration!',
    type=click.Path()
)

def save_parameters_to_json(mtx, dist, calib_file_path) -> None:
    data_dict = {'mtx': mtx.tolist(), 'dist': dist.tolist()}
    with open(calib_file_path, 'w') as output_file:
        json.dump(data_dict, output_file)

def load_parameters_from_json(calib_file_path):
    with open(calib_file_path, 'r') as input_file:
        data_dict = json.load(input_file)
    return np.asarray(data_dict['mtx']), np.asarray(data_dict['dist'])

def calibrate(calib_image_path: Path, calib_file_path: Path) -> None:
    assert str(calib_file_path)[-5:] == '.json', 'Calibration output path has to be a json file.'
    if not os.path.isdir(os.path.split(calib_file_path)[0]):
        os.mkdir(os.path.split(calib_file_path)[0], parents=True)
    assert os.path.isfile(calib_image_path), 'Calibration input image does not exist.'
    objpoints = []
    imgpoints = []
    image = cv2.imread(str(calib_image_path))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    checkerboards_cnt = 0
    for v, h in zip([5,7,5], [15,11,7]):
        ret, corners = cv2.findChessboardCorners(image_gray, (v,h))
        while ret:
            imgpoints.append(corners)
            objp = np.zeros((v*h,3), np.float32)
            objp[:,:2] = np.mgrid[0:v,0:h].T.reshape(-1,2)
            objpoints.append(objp)
            start_x = int(np.min(corners[:,0].T[0]))
            end_x = int(np.max(corners[:,0].T[0]))
            start_y = int(np.min(corners[:,0].T[1]))
            end_y = int(np.max(corners[:,0].T[1]))
            image_gray = cv2.rectangle(image_gray, (start_x, start_y), (end_x, end_y), 0, -1)
            checkerboards_cnt += 1
            ret, corners = cv2.findChessboardCorners(image_gray, (v,h))
    print(checkerboards_cnt, 'checkerboards was found on the image.')
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_gray.shape[::-1], None, None)
    assert ret, 'Undistortion parameters could not be calculated.'
    save_parameters_to_json(mtx, dist, calib_file_path)
    print(str(calib_file_path), 'was created.')

def undistort(image: np.ndarray, calib_file_path: Path):
    assert str(calib_file_path)[-5:] == '.json', 'Calibration file has to be a json file.'
    assert os.path.isfile(calib_file_path), 'Calibration file does not exist.'
    mtx, dist = load_parameters_from_json(calib_file_path)
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
    dst = dst[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
    return dst

if __name__ == '__main__':
    calibrate()