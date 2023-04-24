import click
from rich import print


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


if __name__ == "__main__":
    print()
    print("[bold red]Running video tracking[/bold red]")
    print("[bold red]======================[/bold red]")
    print()
    print("[bold white]Settings[/bold white]")
    run_tracking()
