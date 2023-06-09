import click
import cv2
import numpy as np
from rich import print
import torch
from pathlib import Path
from skimage import io
import pandas as pd
import uuid
from src.tracking.position_calculation import triangulation, project_3d_to_2d_left

from src.tracking.stereo_object_matching import draw_matching_bbox, match_objects
from src.tracking.f2f_object_matching import match_objects_f2f
from src.tracking.kalman_filter import KalmanFilter, KalmanFilterNew


root_dir = Path(__file__).resolve().parents[2]
data_dir = root_dir / "data"
model_dir = root_dir / "models"


@click.command()
@click.option(
    "--seq",
    default=data_dir / "raw/final_project_2023_rect/seq_03",
    prompt="Choose the video sequence: ",
    help="Choose sequence.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
)
# @click.option(
#     "--calib_file",
#     default=data_dir / "processed/calib.json",
#     prompt="Choose calibration file for camera images: ",
#     help="The person to greet.",
#     type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
# )
def run_tracking(seq: Path):
    """Run tracking on video sequence from data folder."""

    print()
    print(f"Sequence path: {seq}")
    print()

    print("[bold blue]Loading model...[/bold blue]")
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=model_dir / "best.pt", force_reload=True
    )
    model.eval()
    print("[bold blue]Model loaded[/bold blue]\n")

    print("[bold blue]Finding frames[/bold blue]")

    dp_left = seq / "image_02/data"
    dp_right = seq / "image_03/data"

    glob_left = sorted(dp_left.glob("*.png"))
    glob_right = sorted(dp_right.glob("*.png"))

    print(glob_left[0])

    print(f"Found {len(glob_left)} frames\n")

    # empty dataframe
    object_columns = [
        "frame",
        "object_id",
        "object_type",
        "observed",
        "position_x",
        "position_y",
        "position_z",
        "last_seen",
        "bbox_cx",
        "bbox_cy",
        "pred_bbox_cx",
        "pred_bbox_cy",
        "bbox_w",
        "bbox_h",
    ]
    objects_df = pd.DataFrame(columns=object_columns)
    zipped_img_list = zip(glob_left, glob_right)
    kf_dict = {}

    for frame, (img_left_path, img_right_path) in enumerate(zipped_img_list):
        print(
            f"[bold white]Processing frame [bold blue]{frame}[/bold blue]: [bold blue]{img_left_path.name}[/bold blue][/bold white]"
        )

        # img_left_path = (
        #     data_dir / "raw/final_project_2023_rect/seq_03/image_02/data/0000000000.png"
        # )  # or file, Path, PIL, OpenCV, numpy, list
        # img_right_path = (
        #     data_dir / "raw/final_project_2023_rect/seq_03/image_03/data/0000000000.png"
        # )  # or file, Path, PIL, OpenCV, numpy, list

        # read images
        img_left = io.imread(img_left_path)
        img_right = io.imread(img_right_path)

        # Inference
        results = model([img_left, img_right])

        # Results
        r_xyxy = results.pandas().xyxy
        r_xywh = results.pandas().xywh

        # results in bottom left corner, top right corner format
        results_left_df = r_xyxy[0]
        results_right_df = r_xyxy[1]

        # match objects
        row_ind, col_ind = match_objects(
            results_left_df, results_right_df, img_left, img_right
        )

        # draw_matching_bbox(
        #     results_left_df, results_right_df, img_left, img_right, row_ind, col_ind
        # )

        # results in center, width, height format
        results_left_df = r_xywh[0]
        results_right_df = r_xywh[1]

        # create new row for each matched object
        for idx_left, idx_right in zip(row_ind, col_ind):
            box_left = results_left_df.iloc[idx_left]
            box_right = results_right_df.iloc[idx_right]

            # calculate position (uses triangulation)
            x, y, z = triangulation(
                box_left["xcenter"],
                box_left["ycenter"],
                box_right["xcenter"],
                box_right["ycenter"],
            )

            # determine id
            # if first frame, create new id
            # else, try to match from previous frame
            if frame == 0:
                obj_id = uuid.uuid4()
            else:
                obj_id = None

            new_row_df = pd.DataFrame(
                [
                    [
                        frame,
                        obj_id,
                        box_left["name"],
                        True,
                        x,
                        y,
                        z,
                        frame,
                        box_left["xcenter"],
                        box_left["ycenter"],
                        0,
                        0,
                        box_left["width"],
                        box_right["height"],
                    ]
                ],
                columns=object_columns,
            )
            objects_df = pd.concat([objects_df, new_row_df], ignore_index=True)

        if frame > 0:
            img_latest = img_left

            for prev_frame_i in range(1, min(frame + 1, 100)):
                prev_frame = frame - prev_frame_i

                # get all rows from the latest frame
                latest_frame_df = objects_df[objects_df["frame"] == frame]

                latest_frame_with_no_uuid_df = latest_frame_df[
                    latest_frame_df["object_id"].isnull()
                ]
                # print(f"{len(latest_frame_with_no_uuid_df)=}")
                if len(latest_frame_with_no_uuid_df) == 0:
                    break

                # get all rows from the previous frame (that were observed)
                prev_frame_df = objects_df[
                    (objects_df["frame"] == prev_frame)
                    & (objects_df["observed"] == True)
                ]

                # get previous frame image
                # img_latest = io.imread(glob_left[frame])
                img_prev = io.imread(glob_left[prev_frame])

                # print(f"{frame=}")
                # print(f"{prev_frame=}")
                # print(f"{glob_left[frame]=}")
                # print(f"{glob_left[prev_frame]=}")
                # io.imsave("test_latest.png", img_latest)
                # io.imsave("test_prev.png", img_prev)

                # match ovjects from previous frame to latest frame
                row_ind, col_ind = match_objects_f2f(
                    latest_frame_df, prev_frame_df, img_latest, img_prev
                )

                # print(row_ind, col_ind)
                for idx_latest, idx_prev in zip(row_ind, col_ind):
                    # print(f"{objects_df.loc[idx_latest, 'object_id']=}")
                    # print(f"{prev_frame_df.loc[idx_prev, 'object_id']=}")

                    id_to_set = prev_frame_df.loc[idx_prev, "object_id"].values[0]
                    # check if id_to_set exist in the latest frame
                    if id_to_set in latest_frame_df["object_id"].values:
                        continue

                    objects_df.loc[idx_latest, "object_id"] = prev_frame_df.loc[
                        idx_prev, "object_id"
                    ].values[0]
                # for idx_latest, idx_prev in zip(row_ind, col_ind):
                #     print(f"{objects_df.loc[idx_latest, 'object_id']=}")
                #     print(f"{prev_frame_df.loc[idx_prev, 'object_id']=}")
            else:
                latest_frame_df = objects_df[objects_df["frame"] == frame]
                latest_frame_with_no_uuid_df = latest_frame_df[
                    latest_frame_df["object_id"].isnull()
                ]
                if len(latest_frame_with_no_uuid_df) > 0:
                    for idx_latest in latest_frame_with_no_uuid_df.index:
                        objects_df.loc[idx_latest, "object_id"] = uuid.uuid4()

            print("took", prev_frame_i, "frames to match all objects")

        # Position prediciton using Kalman filter

        # Find objects in current frame and previous frame
        df_current_frame = objects_df[objects_df["frame"] == frame]
        df_previous_frame = objects_df[objects_df["frame"] == frame - 1]

        # For observed objects, update the kalman filter instance and predict the next position
        for _, row in df_current_frame.iterrows():
            x = row["position_x"]
            y = row["position_y"]
            z = row["position_z"]
            cx = row["bbox_cx"]
            cy = row["bbox_cy"]
            object_id = row["object_id"]

            # create a 9 column vector from the xyz position
            # x_measured = np.array([[x],[y],[z]])

            if object_id in kf_dict.keys():
                kalman_filter = kf_dict[object_id]
            else:
                kalman_filter = KalmanFilterNew(cx, cy)
                kf_dict[object_id] = kalman_filter

            # kalman_filter.update(x_measured)

            ### Predict the next state
            prediction = kalman_filter.predict(cx, cy)

            # Add predicted position to the dataframe
            pred_cx = prediction[0, 0]
            pred_cy = prediction[1, 0]
            # pred_z = kf_dict[object_id].x[2][0]
            objects_df.loc[
                (objects_df["object_id"] == object_id) & (objects_df["frame"] == frame),
                "pred_bbox_cx",
            ] = pred_cx
            objects_df.loc[
                (objects_df["object_id"] == object_id) & (objects_df["frame"] == frame),
                "pred_bbox_cy",
            ] = pred_cy
            # objects_df.loc[(objects_df["object_id"] == object_id) & (objects_df["frame"] == frame), "prediction_z"] = pred_z

        # For non-observed objects,
        # find object ids that are not in the current frame but are in the previous frame
        object_ids_previous_frame = df_previous_frame["object_id"].unique()
        object_ids_current_frame = df_current_frame["object_id"].unique()
        object_ids_non_observed = np.setdiff1d(
            object_ids_previous_frame, object_ids_current_frame
        )

        # predict the next position and use previous predicted position as current position
        rows_to_add = []
        for object_id in object_ids_non_observed:
            assert (
                object_id in kf_dict.keys()
            ), f"Kalman filter does not exist for object ID: {object_id}"
            kalman_filter = kf_dict[object_id]

            # TODO: check if this is correct, do we need to update the kalman filter with the previous predicted position?
            # est_x = kalman_filter.x[0][0]
            # est_y = kalman_filter.x[1][0]
            # est_z = kalman_filter.x[2][0]
            # x_measured = np.array([[est_x],[est_y],[est_z]])
            # kalman_filter.update(x_measured)
            est_cx = df_previous_frame.loc[
                df_previous_frame["object_id"] == object_id, "pred_bbox_cx"
            ].values[0]
            est_cy = df_previous_frame.loc[
                df_previous_frame["object_id"] == object_id, "pred_bbox_cy"
            ].values[0]

            prediction = kalman_filter.predict(est_cx, est_cy)
            # kalman_filter.predict()

            # bbox_cx, bbox_cy = project_3d_to_2d_left([est_x, est_y, est_z])

            # predicted position
            pred_cx = prediction[0, 0]
            pred_cy = prediction[1, 0]
            # pred_z = kf_dict[object_id].x[2][0]

            # Get object type, last seen, bbox width and bbox height from last frame it was observed
            object_type = df_previous_frame[
                df_previous_frame["object_id"] == object_id
            ]["object_type"].values[0]
            last_seen = df_previous_frame[df_previous_frame["object_id"] == object_id][
                "last_seen"
            ].values[0]
            bbox_w = df_previous_frame[df_previous_frame["object_id"] == object_id][
                "bbox_w"
            ].values[0]
            bbox_h = df_previous_frame[df_previous_frame["object_id"] == object_id][
                "bbox_h"
            ].values[0]

            # Make new row for non-observed object in dataframe
            row = [
                frame,
                object_id,
                object_type,
                False,
                0,
                0,
                0,
                last_seen,
                est_cx,
                est_cy,
                pred_cx,
                pred_cy,
                bbox_w,
                bbox_h,
            ]
            rows_to_add.append(row)

        objects_df = pd.concat(
            [objects_df, pd.DataFrame(rows_to_add, columns=object_columns)],
            ignore_index=True,
        )
    objects_df.to_csv("test.csv")


if __name__ == "__main__":
    print()
    print("[bold red]Running video tracking[/bold red]")
    print("[bold red]======================[/bold red]")
    print()
    print("[bold white]Settings[/bold white]")
    run_tracking()
