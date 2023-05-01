import numpy as np

def position_calculation(ul, vl, ur, vr, c_mat=None, frame="camera"):
    """
    Find object position (center of box) wrt Left camera frame (Cam2) or Car frame, returns a 3D position vector
    """


    # Camera intrinsic parameters
    c_mat = c_mat or [[1228.8600498339667, 0.0, 687.0567266957923], [0.0, 1074.1759184503744, 254.13182034905003], [0.0, 0.0, 1.0]]

    fx = c_mat[0,0]
    fy = c_mat[1,1]
    ox = c_mat[0,2]
    oy = c_mat[1,2]

    b = 0.54

    v_avg = (vl + vr)/2

    # Calculate position wrt Cam 2 (left) frame
    z_cam = b*fx/(ur-ul)
    x_cam = z_cam*(ul-ox)/fx
    y_cam = z_cam*(v_avg-oy)/fy
    obj_cam = np.array([x_cam, y_cam, z_cam, 1])


    # Tranformation matrix between Cam 2 (left) and Car frames
    Tcam_car = np.array([[1, 0, 0, 0.06],
                    [0, 1, 0, 1.65],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Position wrt Car frame
    obj_car = np.dot(Tcam_car, obj_cam.T)

    if frame == "camera":
        return obj_cam[0:3]
    elif frame == "car":
        return obj_car[0:3]