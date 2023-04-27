# -*- coding: utf-8 -*-
from typing import Literal
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

from sklearn.model_selection import train_test_split

import shutil

np.random.seed(42)

root_dir = Path(__file__).resolve().parents[2]
data_dir = root_dir / "data"


def create_dirs(out_path: Path):
    (data_dir / "interim").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)

    shutil.rmtree(out_path, ignore_errors=True)
    # (out_path / "images").mkdir(parents=True, exist_ok=True)
    # (out_path / "labels").mkdir(parents=True, exist_ok=True)

    # create dirs for yolo
    (out_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (out_path / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (out_path / "test" / "images").mkdir(parents=True, exist_ok=True)
    (out_path / "test" / "labels").mkdir(parents=True, exist_ok=True)
    (out_path / "val" / "images").mkdir(parents=True, exist_ok=True)
    (out_path / "val" / "labels").mkdir(parents=True, exist_ok=True)


def read_seq(
    seq: str | list[str] | tuple[str],
    src_path: Path,
    data_names: list[str],
    categories: list[str],
) -> pd.DataFrame:
    # check if seq is a list
    if isinstance(seq, str):
        print("seq is not a list")
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
        df["sequence"] = seq
        df = df.reset_index(drop=True)

    else:
        print("seq is a list")
        df = pd.DataFrame()
        for s in seq:
            df_temp = pd.read_csv(
                src_path / f"{s}.txt",
                delimiter=" ",
                names=data_names,
                index_col=False,
            )
            df_temp = df_temp[df_temp["type"].isin(categories)]
            df_temp = df_temp.drop(
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
            df_temp["sequence"] = s
            df_temp = df_temp.reset_index(drop=True)
            # insert df_temp row into df
            df = pd.concat([df, df_temp], ignore_index=True)

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
    categories: list[str],
    src_path_imgs: Path,
    out_path: Path,
    split: Literal["train" , "test" , "val"],
):
    seq = img_name.parent.stem
    save_name = f"{seq}_{img_name.stem}"
    img_path = src_path_imgs / seq / img_name
    img = io.imread(img_path)

    img_height = img.shape[0]
    img_width = img.shape[1]
    io.imsave(out_path / split / "images" / f"{save_name}.jpeg", img)

    df_temp = seq_df[seq_df["frame"].isin([int(img_name.stem)])]

    with (out_path / split / "labels" / f"{save_name}.txt").open(mode="w") as label_file:
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
    print("Making final data set from raw data")
    print(f"Using sequences: {seq}")

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

    # create a list of all the image names in seq
    img_names = []
    for sequence in seq:
        seq_img_names = (src_path_imgs / sequence).glob("*.png")
        # prepend the sequence name to the image name using f-strings
        img_names.extend(seq_img_names)

    # define train val test split percentages
    train_split = 0.6
    val_split = 0.2
    test_split = 0.2

    # split the image names into a train val test split using train_test_split
    img_names_train, img_names_test = train_test_split(
        img_names, test_size=test_split, random_state=42
    )
    img_names_train, img_names_val = train_test_split(
        img_names_train,
        test_size=val_split / (train_split + val_split),
        random_state=42,
    )

    print(f"Number of train images: {len(img_names_train)}")
    print(f"Number of val images: {len(img_names_val)}")
    print(f"Number of test images: {len(img_names_test)}")

    seq_df = read_seq(seq, src_path_labels, data_names, categories)

    for split, img_names_split in zip(["train", "val", "test"], [img_names_train, img_names_val, img_names_test]):
        for img_name in tqdm(
            img_names_split, position=1, desc=split, leave=False, colour="red", ncols=80
        ):
            process_image(seq_df, img_name, categories, src_path_imgs, out_path, split)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
