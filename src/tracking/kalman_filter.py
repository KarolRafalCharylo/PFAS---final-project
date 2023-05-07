import cv2
import numpy as np
import pandas as pd
from src.tracking.position_calculation import project_3d_to_2d_left


class KalmanFilterNew():

    def __init__(self, coordX, coordY):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                        [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        
        # print(self.kf.statePre)
        self.kf.statePre = np.array([[coordX], [coordY], [0], [0]], np.float32)
    
    
    def predict(self, coordX, coordY):
        measured = np.array([[coordX], [coordY]], np.float32)
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted
    
    
class KalmanFilter:
    # generate docstring
    """
    Kalman filter implementation for 3D object tracking
    Implemented from https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CA-Ball.ipynb
    """
    def __init__(self, x0):
        self.dt = 1/30 # time step
        self.x = x0

        # The initial uncertainty (9x9).
        self.P = 1000 * np.identity(9)

        # The external motion (9x1).
        self.u = np.zeros((9,1))

        # The transition matrix (9x9). 
        self.F = np.array([[1.0, 0.0, 0.0, self.dt, 0.0,      0.0,              1/2.0*self.dt**2, 0.0,              0.0],
                           [0.0, 1.0, 0.0, 0.0,     self.dt,  0.0,              0.0,              1/2.0*self.dt**2, 0.0],
                           [0.0, 0.0, 1.0, 0.0,     0.0,      self.dt,          0.0,              0.0,              1/2.0*self.dt**2],
                           [0.0, 0.0, 0.0, 1.0,     0.0,      0.0,              self.dt,          0.0,              0.0],
                           [0.0, 0.0, 0.0, 0.0,     1.0,      0.0,              0.0,              self.dt,          0.0],
                           [0.0, 0.0, 0.0, 0.0,     0.0,      1.0,              0.0,              0.0,              self.dt],
                           [0.0, 0.0, 0.0, 0.0,     0.0,      0.0,              1.0,              0.0,              0.0],
                           [0.0, 0.0, 0.0, 0.0,     0.0,      0.0,              0.0,              1.0,              0.0],
                           [0.0, 0.0, 0.0, 0.0,     0.0,      0.0,              0.0,              0.0,              1.0]])

        # The observation matrix (3x9).
        self.H = np.array([[1,0,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0,0]])

        # The measurement uncertainty.
        self.R = np.identity(3)

        self.I = np.identity(9)


    def update(self, Z):
        y = Z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = (self.P @ self.H.T) @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        self.P = (self.I -  (K @ self.H)) @ self.P
        assert self.x.shape == (9,1)
        # return self.x, self.P

    # def update(self, Z):
    #     y = Z - np.dot(self.H, self.x)
    #     S = np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R
    #     K = np.dot(np.dot(self.P, np.transpose(self.H)), np.linalg.pinv(S))
    #     self.x = self.x + np.dot(K, y)
    #     self.P = np.dot(self.I - np.dot(K, self.H), self.P)
    #     # return self.x, self.P

    def predict(self):
        self.x = self.F @ self.x + self.u
        self.P = self.F @ self.P @ self.F.T
        # return self.x, self.P
    



if __name__ == "__main__":
   
    #read csv file from data folder
    path = "../data/test_final.csv"

    objects_df = pd.read_csv(path)
    objects_df = objects_df.drop(objects_df.columns[0], axis=1)

    kalman_filter_dict = {}


    # test kalman filter iterating to the first 3 frames
    for frame in range(285):

        print(frame)

        # Iterate through dataframe for i corresponding to frame number
        df_current_frame = objects_df[objects_df["frame"] == frame]
        df_previous_frame = objects_df[objects_df["frame"] == frame-1]


        # OBSERVED OBJECTS

        for index, row in df_current_frame.iterrows():
            x = row["position_x"]
            y = row["position_y"]
            z = row["position_z"]
            
            if object_id in kalman_filter_dict.keys():
                kalman_filter = kalman_filter_dict[object_id]

            else: 
                kalman_filter = KalmanFilter()
                kalman_filter_dict[object_id] = kalman_filter

            kalman_filter.update(np.array([x,y,z]))

            ### Predict the next state
            kalman_filter.predict()


        # NON-OBSERVED OBJECTS
        
        # find object ids that are not in the current frame but are in the previous frame
        object_ids_previous_frame = df_previous_frame["object_id"].unique()
        object_ids_current_frame = df_current_frame["object_id"].unique()
        object_ids_non_observed = np.setdiff1d(object_ids_previous_frame, object_ids_current_frame)

        rows_to_make = []

        for object_id in object_ids_non_observed:
            
            kalman_filter = kalman_filter_dict[object_id]

            # ### Predict the next state
            est_x = kalman_filter.x[0][0]
            est_y = kalman_filter.x[0][1]
            est_z = kalman_filter.x[0][2]
            kalman_filter.predict()

            object_type = df_previous_frame[df_previous_frame["object_id"] == object_id]["object_type"].values[0]
            last_seen = df_previous_frame[df_previous_frame["object_id"] == object_id]["last_seen"].values[0]
            bbox_w = df_previous_frame[df_previous_frame["object_id"] == object_id]["bbox_w"].values[0]
            bbox_h = df_previous_frame[df_previous_frame["object_id"] == object_id]["bbox_h"].values[0]
            bbox_cx, bbox_cy = project_3d_to_2d_left([est_x, est_y, est_z])

            # Make new row for non-observed object in dataframe
            row = {
                "object_id": object_id,
                "object_type": object_type,
                "position_x": est_x,
                "position_y": est_y,
                "position_z": est_z,
                "last_seen": last_seen,
                "bbox_cx": bbox_cx,
                "bbox_cy": bbox_cy,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h
                }
            rows_to_make.append(row)
            print(row)

       