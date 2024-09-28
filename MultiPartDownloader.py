import json
import sys
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
from tqdm import tqdm
from worker import Worker

"""
Script to define a class for downloading AudioSet data in parallel
"""

class MultiPartDownloader:

    def __init__(self, num_jobs : int, metadata_df: pd.DataFrame, class_labels_df : pd.DataFrame, download_dir: Path, sleep_amount: int, current_download_info_dir : Path):
        """Main function to initiate parallel downloads with progress tracking."""
        # if not os.path.exists(download_dir):
        #     os.makedirs(download_dir)
        
        self.num_jobs : int = num_jobs
        self.metadata_df : pd.DataFrame = metadata_df
        self.class_labels_df : pd.DataFrame = class_labels_df
        self.download_dir  : Path = Path(download_dir)
        self.sleep_amount : int = sleep_amount
        self.current_download_info : Path = current_download_info_dir

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
    
    def logger(self, worker_processes : list[Worker], total_num_files : int) -> None:
        """Main function to log the progress of the downloads."""
        downloaded_ids : list[str] = []
        errored_ids : list[str] = []

        total_downloaded_files : int = 0
        total_errored_files : int = 0
        with tqdm(total=total_num_files) as pbar:
            while len(downloaded_ids) + len(errored_ids)  < total_num_files:

                last_status : dict = {
                    x.job_id : {

                    } for x in worker_processes
                }

                for w in worker_processes:
                    with w.lock:
                        dict_prox : DictProxy = w.dict_proxy
                        job_ids : list[str] = dict_prox["Downloaded_Ids"]
                        error_ids : list[str] = dict_prox["Error_Ids"]

                        downloaded_ids += job_ids
                        errored_ids += error_ids

                        last_status[w.job_id] = {
                            "NumDownloaded" : len(job_ids),
                            "NumErrored" : len(error_ids),
                            "Total" : len(job_ids) + len(error_ids),
                            "Downloaded" : job_ids,
                            "Errored" : error_ids,
                            "UpdateTimestamp" : time.time()
                        }

                with open(self.current_download_info / Path("download_status.json"), "w") as f:
                    last_status["TotalNumDownloaded"] = len(downloaded_ids)
                    last_status["TotalNumErrored"] = len(errored_ids)
                    last_status["command"] = str(sys.executable) + " " + " ".join(sys.argv)
                    json.dump(last_status, f, indent=4)
                
                with open(self.current_download_info / Path("split_current_ytids.txt"), "w") as f:
                    f.write("\n".join(downloaded_ids))

                # print(f"{len(downloaded_ids)=}")   
                pbar.update(len(downloaded_ids) + len(errored_ids))
                # print(f"{len(downloaded_ids) + len(errored_ids)=}")
                pbar.set_postfix({"Downloaded" : len(downloaded_ids), "Errored" : len(errored_ids)})
                time.sleep(2)
    
    def download_yt_row(self, index : int, meta_row : Series, dict : DictProxy, job_id : int, lock : mp.Lock) -> None:
        """Downloads a file from a URL and saves it to the specified directory."""

        ytid : str = meta_row['YTID']
        start_time : str = meta_row['start_seconds']
        end_time : str = meta_row['end_seconds']
        positive_labels : list[str] = [self.machine_to_display_mapping[x] for x in meta_row['positive_labels'].split(",")]

        download_paths : list[Path] = [Path(self.download_dir / label / Path(f"{ytid}_{start_time}-{end_time}.wav")) for label in positive_labels]
        # print(f"Downloading {ytid} from {start_time} to {end_time} to {download_paths}")

        ret_pair : tuple[int, Optional[Exception]] = download_audio_section(ytid, start_time, end_time, download_paths , "wav", True)

        assert ret_pair is not None, f"Error downloading {ytid} from {start_time} to {end_time}"

        # print(f"{ret_pair=}")
        # while we still keep getting bot sniped, try again
        while ret_pair[0] == 1 and "Sign in to confirm you\u2019re not a bot. This helps protect our community. Learn more" in str(ret_pair[1]):
            print(f"Job number {job_id} got bot sniped. Retrying download for ytid={ytid} after sleeping for {self.sleep_amount} seconds")
            time.sleep(self.sleep_amount)
            ret_pair = download_audio_section(ytid, start_time, end_time, download_paths, "wav", True)
        
        assert ret_pair is not None, f"Error downloading {ytid} from {start_time} to {end_time}"
        
        with lock:
            if ret_pair[0] == 1 or ret_pair[1] is not None:
                dict["Errors"].append(ret_pair[1])
                dict["Error_Ids"].append(ytid)
            elif ret_pair[0] == 0:
                dict["Downloaded_Ids"].append(ytid)

    def init_multipart_download(self):

        with mp.Manager() as manager:

            # Split URLs across workers

            metadata_list : list[pd.DataFrame] = self.split_ids(self.metadata_df, self.num_jobs)
            print(f"{metadata_list=}")

            worker_processes : list[Worker] = []

            for job_id, meta_slice in enumerate(metadata_list):

                dict_prox : DictProxy = manager.dict()
                dict_prox["Errors"] = []
                dict_prox["Error_Ids"] = []
                dict_prox["Downloaded_Ids"] = []
                lock : mp.Lock = manager.Lock()
                process : mp.Process = mp.Process(target=self.worker, args=(meta_slice, dict_prox, job_id, lock))

                worker_processes.append(Worker(dict_prox, lock, job_id, process))

            logger_process : mp.Process = mp.Process(target=self.logger, args=(worker_processes, len(self.metadata_df)))
            logger_process.start()
            
            # Start worker processes
            for i, w in enumerate(worker_processes):
                print(f"Starting process {i}")
                w.process.start()

            print("Started all processes")
            # Wait for all workers to finish
            for w in worker_processes:
                w.process.join()

            logger_process.close()