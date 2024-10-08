from argparse import ArgumentParser, Namespace
import time
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


def get_existing_ytids(split_dir: Path) -> set[str]:
    print(f"Checking how many files are under {split_dir}")

    command: str = f"ls -1r {split_dir} | wc -l"
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
    command: str = f"ls -1R {split_dir} | grep -o -P '.*(?=_\d+\.\d+[-_]\d+\.\d.wav)'"
    result: CompletedProcess = subprocess.run(
        command, shell=True, capture_output=True, text=True
    )
    assert isinstance(result, CompletedProcess)

    assert result.returncode == 0, "Was unable to determine the IDs of the files"
    # print(f"{result=}")

    # Get the output as a string
    output: str = result.stdout

    # Split the output into lines
    existing_ytids: list[str] = set(output.strip().split("\n"))
    # print(f"{existing_ytids=}")
    print(f"Found {len(existing_ytids)} existing wav files under {split_dir}.")
    return existing_ytids


def get_excluded_ytids(exclusion_ids_file) -> list[str]:
    if not exclusion_ids_file.exists():
        print(f"Creating {exclusion_ids_file}")
        exclusion_ids_file.touch()

    with open(str(exclusion_ids_file), "r") as f:
        excluded_files: list[str] = f.readlines()
    return excluded_files

def args_checks(args : Namespace):
    assert args.split in split_names, f"Invalid split name: {args.split}"
    assert args.n_splits >= 1, "Number of splits must be at least 1"
    assert args.split_idx >= 0, "Split index must be at least 0"
    assert args.n_jobs >= 1, "Number of jobs must be at least 1"
    assert args.sleep_amount >= 0, "Sleep amount must be at least 0"
    assert (
        args.split_idx < args.n_splits
    ), "Split index must be less than the number of splits"

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

    # performs sanity checks for our arguments
    args_checks(args)

    if args.debug:
        print(f"args={args}")
        PORT: int = 5678
        import debugpy

        debugpy.listen(("localhost", PORT))
        debugpy.wait_for_client()
        print(f"Waiting for debugger to attach on {PORT}")

    data_dir : Path = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    split_dir : Path = data_dir / Path(args.split)
    split_dir.mkdir(exist_ok=True)

    cache_dir : Path = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    csvDownloader: CsvDownloader = CsvDownloader(args.split, args.cache_dir)
    print("Loading CSVs...")
    
    # Load the metadata and class mapping CSV
    # for the metadata, only load the rows of the split that we need
    split_df: DataFrame = csvDownloader.load_segment_csv_url(args.n_splits, args.split_idx, CsvDownloader.split_quantities[args.split])
    class_mapping_df: DataFrame = csvDownloader.load_class_mapping_csv()

    print("Fetching existing IDs...")
    existing_ytids: list[str] = get_existing_ytids(split_dir)

    # make the download info dir 
    current_download_info_dir.mkdir(exist_ok=True)

    # Read the exclusions file
    exclusion_ids_file: Path = Path(args.exclusion_ids_file)

    excluded_files = get_excluded_ytids(exclusion_ids_file)
    
    print(f"Found {len(excluded_files)} file exclusions")

    # write a file with the ids to download 

    print(f"Current split index is {args.split_idx} out of {args.n_splits}")
    print(f"Proceeding to download {len(split_df)} files")

    with open(current_download_info_dir / Path("split_total_ytids.txt"), "w") as f:
        f.write("\n".join(split_df["YTID"]))
    
    with open(current_download_info_dir / Path("split_current_ytids.txt"), "w") as f:
        f.write("\n".join(existing_ytids))
    
    print("Filtering out for already downloaded YTIds")

    # # already_downloaded_from_split : list[str] = chosen_df["YTID"]()
    filtered_split_df : pd.DataFrame = split_df[~split_df["YTID"].isin(existing_ytids) & ~split_df["YTID"].isin(excluded_files)]
    # assert len(filtered_split_df) + len(existing_ytids) + len(excluded_files) == len(split_df), "Length of filtered dataframe plus existing files should sum up to original length of split dataframe"

    # print(f"Found {len(excluded_files)} file exclusions")
    # print(f"Downloading {len(filtered_split_df)} files")
    # self, num_jobs : int, metadata_df: pd.DataFrame, class_labels_df : pd.DataFrame, download_dir: Path, sleep_amount: int
    multi_part_downloader: MultiPartDownloader = MultiPartDownloader(
        args.n_jobs, filtered_split_df, class_mapping_df, split_dir,
        args.sleep_amount, current_download_info_dir, len(split_df), list(excluded_files), list(existing_ytids)
    )

    multi_part_downloader.init_multipart_download()
