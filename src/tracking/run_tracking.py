import click
import cv2
from rich import print
import torch
from pathlib import Path
from skimage import io

root_dir = Path(__file__).resolve().parents[2]
data_dir = root_dir / 'data'
model_dir = root_dir / 'models'

@click.command()
@click.option(
    "--seq",
    default="raw/final_project/seq_03",
    prompt="Choose the video sequence: ",
    help="Choose sequence.",
)
@click.option(
    "--calib_file",
    default="processed/calib.json",
    prompt="Choose calibration file for camera images: ",
    help="The person to greet.",
)
def run_tracking(seq, calib_file):
    """Run tracking on video sequence from data folder."""

    print()
    print(f"Sequence path: {seq}")
    print(f"Calibration file: {calib_file}")

    print("[bold blue]Loading model...[/bold blue]")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_dir / 'best.pt', force_reload=True)
    model.eval()
    print("[bold blue]Model loaded[/bold blue]")

    img = data_dir / "raw/final_project/seq_03/image_02/data/0000000000.png"  # or file, Path, PIL, OpenCV, numpy, list

    # Inference
    results = model(img)

    # Results
    r = results.pandas().xyxy

    frame = io.imread(img)
    for _, box in r[0].iterrows():
        if box["class"]==0:
            xB = int(box["xmin"])
            yB = int(box["ymin"])
            xA = int(box["xmax"])
            yA = int(box["ymax"])
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    io.imsave("test.png", frame)




if __name__ == "__main__":
    print()
    print("[bold red]Running video tracking[/bold red]")
    print("[bold red]======================[/bold red]")
    print()
    print("[bold white]Settings[/bold white]")
    run_tracking()
