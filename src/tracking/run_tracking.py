import click
import cv2
import numpy as np
from rich import print
import torch
from pathlib import Path
from skimage import io
import pandas as pd
from src.tracking.position_calculation import position_calculation

from src.tracking.stereo_object_matching import draw_matching_bbox, match_objects

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
        "position_x",
        "position_y",
        "position_z",
        "last_seen",
        "bbox_cx",
        "bbox_cy",
        "bbox_w",
        "bbox_h",
    ]
    objects_df = pd.DataFrame(columns=object_columns)
    for frame, (img_left_path, img_right_path) in enumerate(zip(glob_left, glob_right)):
        print(f"[bold white]Processing frame [bold blue]{frame}[/bold blue]: [bold blue]{img_left_path.name}[/bold blue][/bold white]")

        # img_left_path = (
        #     data_dir / "raw/final_project_2023_rect/seq_03/image_02/data/0000000000.png"
        # )  # or file, Path, PIL, OpenCV, numpy, list
        # img_right_path = (
        #     data_dir / "raw/final_project_2023_rect/seq_03/image_03/data/0000000000.png"
        # )  # or file, Path, PIL, OpenCV, numpy, list

        img_left = io.imread(img_left_path)
        img_right = io.imread(img_right_path)



        # Inference
        results = model([img_left, img_right])

        # Results
        r_xyxy = results.pandas().xyxy
        r_xywh = results.pandas().xywh

        results_left_df = r_xyxy[0]
        results_right_df = r_xyxy[1]

        row_ind, col_ind = match_objects(
            results_left_df, results_right_df, img_left, img_right
        )

        draw_matching_bbox(
            results_left_df, results_right_df, img_left, img_right, row_ind, col_ind
        )

        results_left_df = r_xywh[0]
        results_right_df = r_xywh[1]

        for idx_left, idx_right in zip(row_ind, col_ind):
            box_left = results_left_df.iloc[idx_left]
            box_right = results_right_df.iloc[idx_right]

            x, y, z = position_calculation(
                box_left["xcenter"],
                box_left["ycenter"],
                box_right["xcenter"],
                box_right["ycenter"],
            )
            obj_id = np.random.randint(20)
            new_row_df = pd.DataFrame(
                [
                    [
                        frame,
                        obj_id,
                        box_left["name"],
                        x,
                        y,
                        z,
                        0,
                        box_left["xcenter"],
                        box_left["ycenter"],
                        box_left["width"],
                        box_right["height"],
                    ]
                ],
                columns=object_columns,
            )
            objects_df = pd.concat([objects_df, new_row_df], ignore_index=True)
            pass
        objects_df.to_csv("test.csv")


if __name__ == "__main__":
    print()
    print("[bold red]Running video tracking[/bold red]")
    print("[bold red]======================[/bold red]")
    print()
    print("[bold white]Settings[/bold white]")
    run_tracking()
