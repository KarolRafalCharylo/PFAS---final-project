# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from pathlib import Path
from tqdm import tqdm
import numpy as np
import seaborn as sns
from pylab import rcParams
import pandas as pd
from skimage import io

import shutil

np.random.seed(42)

root_dir = Path(__file__).resolve().parents[2]
data_dir = root_dir / "data"


def create_dirs(out_path: Path):
    (data_dir / "interim").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)

    shutil.rmtree(out_path, ignore_errors=True)
    (out_path / "images").mkdir(parents=True, exist_ok=True)
    (out_path / "labels").mkdir(parents=True, exist_ok=True)


def read_seq(
    seq: str, src_path: Path, data_names: list[str], categories: list[str]
) -> pd.DataFrame:
    df = pd.read_csv(
        src_path / f"{seq}.txt", delimiter=" ", names=data_names, index_col=False
    )
    df = df[df["type"].isin(categories)]
    df = df.drop(
        columns=[
            "truncated",
            "occluded",
            "alpha",
            "dimension_width",
            "dimension_height",
            "dimension_length",
            "rotation_y",
        ]
    )
    df = df.reset_index(drop=True)
    return df


def format_yolo(
    df: pd.DataFrame, img_width: int, img_height: int, categories: list[str]
) -> pd.DataFrame:
    df["category_idx"] = categories.index(df["type"])

    df["bbox_width"] = df["bbox_right"] - df["bbox_left"]
    df["bbox_height"] = df["bbox_bottom"] - df["bbox_top"]

    df["bbox_x"] = df["bbox_left"] + df["bbox_width"] / 2
    df["bbox_y"] = df["bbox_top"] + df["bbox_height"] / 2
    df["bbox_x"] = df["bbox_x"] / img_width
    df["bbox_y"] = df["bbox_y"] / img_height
    df["bbox_width"] = df["bbox_width"] / img_width
    df["bbox_height"] = df["bbox_height"] / img_height
    df = df.drop(columns=["bbox_left", "bbox_top", "bbox_right", "bbox_bottom"])

    return df


def process_image(
    seq_df: pd.DataFrame,
    img_name: Path,
    seq: str,
    categories: list[str],
    src_path_imgs: Path,
    out_path: Path,
):
    save_name = f"{seq}_{img_name.stem}"
    img_path = src_path_imgs / seq / img_name
    img = io.imread(img_path)

    img_height = img.shape[0]
    img_width = img.shape[1]
    io.imsave(out_path / "images" / f"{save_name}.jpeg", img)

    df_temp = seq_df[seq_df["frame"].isin([int(img_name.stem)])]

    with (out_path / "labels" / f"{save_name}.txt").open(mode="w") as label_file:
        for _, row in df_temp.iterrows():
            category_idx = categories.index(row["type"])

            x1 = row["bbox_left"] / img_width
            y1 = row["bbox_top"] / img_height
            x2 = row["bbox_right"] / img_width
            y2 = row["bbox_bottom"] / img_height

            bbox_width = x2 - x1
            bbox_height = y2 - y1

            label_file.write(
                f"{category_idx} {x1 + bbox_width / 2} {y1 + bbox_height / 2} {bbox_width} {bbox_height}\n"
            )


@click.command()
@click.option("--seq", default=["0000"], help="Choose sequences.", multiple=True)
@click.option(
    "--src_path_imgs",
    default="raw/KITTI/data_tracking_image_2/training/label_02/",
    prompt="Choose path of directory where sequences are found: ",
)
@click.option(
    "--src_path_labels",
    default="raw/KITTI/data_tracking_label_2/training/label_02/",
    prompt="Choose path of directory where labels are found: ",
)
@click.option(
    "--out_path",
    default="processed/KITTI/",
    prompt="Choose path of formatted data: ",
)
def main(seq: list[str], src_path_imgs: Path, src_path_labels: Path, out_path: Path):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    print(seq)


    src_path_imgs = data_dir / Path(src_path_imgs)
    src_path_labels = data_dir / Path(src_path_labels)
    out_path = data_dir / Path(out_path)
    create_dirs(out_path)

    data_names = [
        "frame",
        "track_id",
        "type",
        "truncated",
        "occluded",
        "alpha",
        "bbox_left",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
        "dimension_width",
        "dimension_height",
        "dimension_length",
        "location_x",
        "location_y",
        "location_z",
        "rotation_y",
    ]

    categories = ["Car", "Pedestrian", "Cyclist"]

    for sequence in tqdm(seq, position=0, desc="seq", leave=False, colour='green', ncols=80):
        # print(sequence)
        seq_df = read_seq(sequence, src_path_labels, data_names, categories)
        img_names = (src_path_imgs / sequence).glob("*.png")

        for img_name in tqdm(img_names, position=1, desc="img", leave=False, colour='red', ncols=80):
            process_image(
                seq_df, img_name, sequence, categories, src_path_imgs, out_path
            )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
