import numpy as np
from pathlib import Path
import pandas as pd
from typing import Optional
from pandas.core.series import Series
import os
from ytdlp_download import download_audio_section
import time
from multiprocessing.managers import DictProxy
import multiprocessing as mp

"""
Script to define a class for downloading AudioSet data in parallel
"""

class MultiPartDownloader:

    def __init__(self, num_jobs : int, metadata_df: pd.DataFrame, class_labels_df : pd.DataFrame, download_dir: Path, sleep_amount: int):
        """Main function to initiate parallel downloads with progress tracking."""
        # if not os.path.exists(download_dir):
        #     os.makedirs(download_dir)
        
        self.num_jobs : int = num_jobs
        self.metadata_df : pd.DataFrame = metadata_df
        self.class_labels_df : pd.DataFrame = class_labels_df
        self.download_dir  : Path = Path(download_dir)
        self.sleep_amount : int = sleep_amount

        # since each row in the metdata_df object has a set of positive labels
        # with the machine labels, we have to use this to map the machine labels
        # to the display labels
        self.display_to_machine_mapping = dict(
            zip(self.class_labels_df["display_name"], self.class_labels_df["mid"])
        )
        self.machine_to_display_mapping = dict(
            zip(self.class_labels_df["mid"], self.class_labels_df["display_name"])
        )

    def split_ids(self, metadata_df: pd.DataFrame, n_splits: int) -> list[pd.DataFrame]:
        return np.array_split(metadata_df, n_splits)
    
    def worker(self, meta_df : pd.DataFrame, dict : DictProxy, job_id : int, lock : mp.Lock) -> None:
        """Main function to download a set of rows from the metadata CSV."""
        for index, row in meta_df.iterrows():
            self.download_yt_row(index, row, dict, job_id, lock)

    def percentage_fmt(num: float) -> str:
        return "{:.2%}".format(num)
    
    def download_yt_row(self, index : int, meta_row : Series, dict : DictProxy, job_id : int, lock : mp.Lock) -> None:
        """Downloads a file from a URL and saves it to the specified directory."""

        ytid : str = meta_row['YTID']
        start_time : str = meta_row['start_seconds']
        end_time : str = meta_row['end_seconds']
        positive_labels : list[str] = [self.machine_to_display_mapping[x] for x in meta_row['positive_labels'].split(",")]

        download_paths : list[Path] = [Path(self.download_dir / label / Path(f"{ytid}_{start_time}-{end_time}.wav")) for label in positive_labels]
        print(f"Downloading {ytid} from {start_time} to {end_time} to {download_paths}")

        ret_pair : tuple[int, Optional[Exception]] = download_audio_section(ytid, start_time, end_time, download_paths , "wav", True)

        assert ret_pair is not None, f"Error downloading {ytid} from {start_time} to {end_time}"

        print(f"{ret_pair=}")
        # while we still keep getting bot sniped, try again
        while ret_pair == 1 and ("Sign in to confirm you\u2019re not a bot. This helps protect our community. Learn more" in ret_pair[1]):
            print(f"Job number {job_id} got bot sniped. Retrying download for ytid={ytid} after sleeping for {self.sleep_amount} seconds")
            time.sleep(self.sleep_amount)
            ret_pair = download_audio_section(ytid, start_time, end_time, download_paths, "wav", True)
        
        assert ret_pair is not None, f"Error downloading {ytid} from {start_time} to {end_time}"
        
        with lock:
            if(dict.get(job_id) is None):
                dict[job_id] = {
                    "Errors" : [],
                    "Error_Ids" : [],
                    "Downloaded_Ids" : []
                }

            if ret_pair[1] == 1:
                dict[job_id]["Errors"].append(ret_pair[1])
                dict[job_id]["Error_Ids"].append(ytid)
            elif ret_pair[1] == 0:
                dict[job_id]["Downloaded_Ids"].append(ytid)

    def init_multipart_download(self):

        with mp.Manager() as manager:

            dict_prox : DictProxy = manager.dict()
            lock : mp.Lock = manager.Lock()

            # listener = Process(target=progress_listener, args=(queue, len(urls)))
            # listener.start()

            # Split URLs across workers

            metadata_list : list[pd.DataFrame] = self.split_ids(self.metadata_df, self.num_jobs)
            print(f"{metadata_list=}")

            processes = [
                mp.Process(target=self.worker, args=(meta_slice, dict_prox, job_id, lock))
                for job_id, meta_slice in enumerate(metadata_list)
            ]
            
            # Start worker processes
            for i, p in enumerate(processes):
                print(f"Starting process {i}")
                p.start()

            print("Started all processes")
            # Wait for all workers to finish
            for p in processes:
                p.join()