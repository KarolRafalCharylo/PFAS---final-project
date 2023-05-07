import numpy as np


def triangulation(
    ul, vl, ur, vr, c_mat: np.ndarray | None = None, frame="camera"
):
    """
    Find object position (center of box) wrt Left camera frame (Cam2) or Car frame, returns a 3D position vector
    """

    # Camera intrinsic parameters
    c_mat = c_mat or np.array(
        [
            [1228.8600498339667, 0.0, 687.0567266957923],
            [0.0, 1074.1759184503744, 254.13182034905003],
            [0.0, 0.0, 1.0],
        ]
    )

    fx = c_mat[0, 0]
    fy = c_mat[1, 1]
    ox = c_mat[0, 2]
    oy = c_mat[1, 2]

    b = 0.54

    v_avg = (vl + vr) / 2

    # Calculate position wrt Cam 2 (left) frame
    z_cam = b * (-fx) / (ur - ul)
    x_cam = z_cam * (ul - ox) / (-fx)
    y_cam = z_cam * (v_avg - oy) / (-fy)
    obj_cam = np.array([x_cam, y_cam, z_cam, 1])

    # Tranformation matrix between Cam 2 (left) and Car frames
    Tcam_car = np.array([[1, 0, 0, 0.06], [0, 1, 0, 1.65], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Position wrt Car frame
    obj_car = np.dot(Tcam_car, obj_cam.T)

    if frame == "camera":
        return obj_cam[0:3]
    elif frame == "car":
        return obj_car[0:3]
    

def project_3d_to_2d_left(point, c_mat: np.ndarray | None = None):
    """
    Project a 3D point in camera frame to 2D pixel coordinates in left image plane
    """
    c_mat = c_mat or np.array(
        [
            [1228.8600498339667, 0.0, 687.0567266957923],
            [0.0, 1074.1759184503744, 254.13182034905003],
            [0.0, 0.0, 1.0],
        ]
    )

    fx = c_mat[0, 0]
    fy = c_mat[1, 1]
    ox = c_mat[0, 2]
    oy = c_mat[1, 2]

    x = point[0]
    y = point[1]
    z = point[2]

    u = (-fx) * x / z + ox
    v = (-fy) * y / z + oy

    return u, v
    

def bbox_center(bbox):
    """
    Calculate the pixel coordinates of center of a bounding box
    """
    u = bbox[0] + (bbox[2]-bbox[0])/2
    v = bbox[1] + (bbox[3]-bbox[1])/2
    return u, v


if __name__ == "__main__":
    
    bboxes1 = np.array([[0,201.514206,297.433075,370],
                        [1000.916809,151.643967,1076.956909,295.646484],
                        [446.264679,172.997131,483.641022,198.633377],
                        [867.967407,167.970062,917.901123,276.058960]])
    bboxes2 = np.array([[0,195.325424,233.257751,367.463257],
                        [958.674683,145.034378,1032.881836,297.623138],
                        [438.047150,173.084274,475.409637,199.053802],
                        [788.372253,153.427689,893.130249,277.143463]])


    for bbox1, bbox2 in zip(bboxes1, bboxes2):
        ul, vl = bbox_center(bbox1)
        ur, vr = bbox_center(bbox2)
        print(triangulation(ul, vl, ur, vr))


    points_3D = [[ 9.05967229, -0.56686293, 20.68036079],[-4.40272098,  0.45295617, 15.37548564],[14.58285856,  5.12191251, 80.68428431],[-2.13046089,  0.42004612, 12.71648332]]

    new_points_2D = []
    for point in points_3D:
        u, v = project_3d_to_2d_left(point)
        new_points_2D.append([u, v])
        print(u, v)

    for point, bbox1 in zip(new_points_2D, bboxes1):
        ul, vl = bbox_center(bbox1)
        print(point, [ul, vl])