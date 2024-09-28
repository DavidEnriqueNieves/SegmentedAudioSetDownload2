from argparse import ArgumentParser, Namespace
import yt_dlp
import subprocess
from subprocess import CompletedProcess
from csv_setup import CsvDownloader
import pandas as pd
from pandas import DataFrame
from MultiPartDownloader import MultiPartDownloader
from pathlib import Path
import numpy as np

"""
A script made because I can't trust Open Source Code and I'm incompetent
Based off another failed script that I made
"""

split_names: list[str] = [
    "eval_segments",
    "balanced_train_segments",
    "unbalanced_train_segments",
]
current_download_info_dir: Path = Path("./current_download_info")


def get_existing_ytids(data_dir: str) -> list[str]:
    print(f"Checking how many files are under {data_dir}")

    command: str = f"ls -1r {data_dir} | wc -l"
    result: CompletedProcess = subprocess.run(
        command, shell=True, capture_output=True, text=True
    )

    output: str = result.stdout
    # print(f"{output=}")

    if int(output) == 0:
        print("No existing wav files found.")
        return []

    # Execute the command and capture the output
    # grep command to only fetch the IDs of the file
    # matches only the ID part of any file with this format: "ID_dd.d-dd.d.wav"
    command: str = f"ls -1R {data_dir} | grep -o -P '.*(?=_\d+\.\d+[-_]\d+\.\d.wav)'"
    result: CompletedProcess = subprocess.run(
        command, shell=True, capture_output=True, text=True
    )
    assert isinstance(result, CompletedProcess)

    assert result.returncode == 0, "Was unable to determine the IDs of the files"
    # print(f"{result=}")

    # Get the output as a string
    output: str = result.stdout

    # Split the output into lines
    existing_ytids: list[str] = output.strip().split("\n")
    # print(f"{existing_ytids=}")
    print(f"Found {len(existing_ytids)} existing wav files under {data_dir}.")
    return existing_ytids


if __name__ == "__main__":

    argparser: ArgumentParser = ArgumentParser()

    argparser.add_argument( "--data_dir", type=str, required=True, help="Directory to store the downloaded CSV metadata files",)
    argparser.add_argument( "--n_splits", type=int, required=True, help="The number of splits to download")
    argparser.add_argument( "--split_idx", type=int, required=True, help="The index of the split to download",)
    argparser.add_argument( "--n_jobs", type=int, default=1, help="Number of jobs to run in parallel")
    argparser.add_argument( "--sleep_amount", type=int, default=10, help="Amount of time to sleep between downloads",)
    argparser.add_argument( "--split", type=str, required=True, choices=split_names, help="The split to download, one of 'eval_segments', 'balanced_train_segments', 'unbalanced_train_segments'",)
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument( "--cache_dir", type=str, default="./cache", help="Directory to store the downloaded CSV metadata files",)
    argparser.add_argument( "--exclusion_ids_file", type=str, default=f"{current_download_info_dir}/unavailable_ids.txt", help="File containing the IDs of the videos that are unavailable",)

    args: Namespace = argparser.parse_args()

    assert args.split in split_names, f"Invalid split name: {args.split}"
    assert args.n_splits >= 1, "Number of splits must be at least 1"
    assert args.split_idx >= 0, "Split index must be at least 0"
    assert args.n_jobs >= 1, "Number of jobs must be at least 1"
    assert args.sleep_amount >= 0, "Sleep amount must be at least 0"
    assert (
        args.split_idx < args.n_splits
    ), "Split index must be less than the number of splits"

    if args.debug:
        print(f"args={args}")
        PORT: int = 5678
        import debugpy

        debugpy.listen(("localhost", PORT))
        debugpy.wait_for_client()
        print(f"Waiting for debugger to attach on {PORT}")

    csvDownloader: CsvDownloader = CsvDownloader(args.split, args.cache_dir)
    print("Loading CSVs...")
    meta_df: DataFrame = csvDownloader.load_segment_csv_url()
    class_mapping_df: DataFrame = csvDownloader.load_class_mapping_csv()

    print("Fetching existing IDs...")
    existing_ytids: list[str] = get_existing_ytids(args.data_dir)

    # Read the exclusions file

    exclusion_ids_file: Path = Path(args.exclusion_ids_file)

    if not exclusion_ids_file.exists():
        print(f"Creating {exclusion_ids_file}")
        exclusion_ids_file.touch()

    with open(str(exclusion_ids_file), "r") as f:
        exclusions: list[str] = f.readlines()

    # split meta_df into n_splits
    meta_splits : list[DataFrame] = np.array_split(meta_df, args.n_splits)
    chosen_df : DataFrame = meta_splits[args.split_idx]

    # write a file with the ids to download 

    # make the download info dir 
    current_download_info_dir.mkdir(exist_ok=True)

    print(f"Current split index is {args.split_idx} out of {args.n_splits}")
    print(f"Proceeding to download {len(chosen_df)} files")

    with open(current_download_info_dir / Path("split_total_ytids.txt"), "w") as f:
        f.write("\n".join(chosen_df["YTID"]))
    
    with open(current_download_info_dir / Path("split_current_ytids.txt"), "w") as f:
        f.write("")
    
    print(f"Found {len(exclusions)} file exclusions")
    # self, num_jobs : int, metadata_df: pd.DataFrame, class_labels_df : pd.DataFrame, download_dir: Path, sleep_amount: int
    multi_part_downloader: MultiPartDownloader = MultiPartDownloader(
        args.n_jobs, meta_df, class_mapping_df, args.data_dir,
        args.sleep_amount, current_download_info_dir
    )

    multi_part_downloader.init_multipart_download()
