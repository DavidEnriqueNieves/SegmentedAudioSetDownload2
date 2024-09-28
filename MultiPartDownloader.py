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

    def __init__(
        self,
        num_jobs: int,
        filtered_split_df: pd.DataFrame,
        class_labels_df: pd.DataFrame,
        download_dir: Path,
        sleep_amount: int,
        current_download_info_dir: Path,
        total_num_files: int,
        existing_files: list[str],
        excluded_files: list[str],
    ):
        """Main function to initiate parallel downloads with progress tracking."""
        # if not os.path.exists(download_dir):
        #     os.makedirs(download_dir)

        self.num_jobs: int = num_jobs
        self.filtered_split_df: pd.DataFrame = filtered_split_df
        self.class_labels_df: pd.DataFrame = class_labels_df
        self.download_dir: Path = Path(download_dir)
        self.sleep_amount: int = sleep_amount
        self.current_download_info: Path = current_download_info_dir
        self.start_existing_files: list[str] = existing_files
        self.start_excluded_files: list[str] = excluded_files
        self.total_num_files: int = total_num_files

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

    def worker(
        self, meta_df: pd.DataFrame, dict: DictProxy, job_id: int, lock: mp.Lock
    ) -> None:
        """Main function to download a set of rows from the metadata CSV."""
        for index, row in meta_df.iterrows():
            self.download_yt_row(index, row, dict, job_id, lock)

    def percentage_fmt(num: float) -> str:
        return "{:.2%}".format(num)

    def logger(self, worker_processes: list[Worker], total_num_files: int) -> None:
        """Main function to log the progress of the downloads."""
        downloaded_ids: list[str] = []
        errored_ids: list[str] = []

        with tqdm(total=total_num_files) as pbar:
            while len(downloaded_ids) + len(errored_ids) < total_num_files:
                downloaded_ids = []
                errored_ids = []

                last_status: dict = {x.job_id: {} for x in worker_processes}

                for w in worker_processes:
                    with w.lock:
                        dict_prox: DictProxy = w.dict_proxy

                        job_download_ids: list[str] = dict_prox["Downloaded_Ids"]
                        job_error_ids: list[str] = dict_prox["Error_Ids"]
                        # contains error messages for each job
                        job_errors: list[str] = dict_prox["Errors"]

                        downloaded_ids.extend(job_download_ids)
                        errored_ids.extend(job_error_ids)

                        last_status[w.job_id] = {
                            "NumDownloaded": len(job_download_ids),
                            "NumErrored": len(job_error_ids),
                            "Total": len(job_download_ids) + len(job_error_ids),
                            "Downloaded": job_download_ids,
                            "Errored": job_error_ids,
                            "Errors": job_errors,
                            "UpdateTimestamp": time.time(),
                        }

                with open( self.current_download_info / Path("download_status.json"), "w") as f:
                    last_status["TotalNumDownloaded"] = len(downloaded_ids) + len( self.start_existing_files)
                    last_status["TotalNumErrored"] = len(errored_ids) + len( self.start_excluded_files)

                    last_status["StartExistingFiles"] = self.start_existing_files
                    last_status["StartExcludedFiles"] = self.start_excluded_files

                    last_status["command"] = ( str(sys.executable) + " " + " ".join(sys.argv))
                    json.dump(last_status, f, indent=4)

                with open( self.current_download_info / Path("split_current_ytids.txt"), "w") as f:
                    f.write("\n".join(downloaded_ids))

                with open( self.current_download_info / Path("unavailable_ids.txt"), "w") as f:
                    f.write("\n".join(self.start_excluded_files) + "\n".join(errored_ids))

                # print(f"{len(downloaded_ids)=}")
                pbar.n = len(downloaded_ids) + len(errored_ids) + len(self.start_existing_files) + len(self.start_excluded_files)
                pbar.refresh()
                
                # print(f"{len(downloaded_ids) + len(errored_ids)=}")
                pbar.set_postfix(
                    {"Downloaded": len(downloaded_ids) + len(self.start_existing_files), "Errored": len(errored_ids)}
                )
                time.sleep(1)

    def download_yt_row(
        self, index: int, meta_row: Series, dict: DictProxy, job_id: int, lock: mp.Lock
    ) -> None:
        """Downloads a file from a URL and saves it to the specified directory."""

        ytid: str = meta_row["YTID"]
        start_time: str = meta_row["start_seconds"]
        end_time: str = meta_row["end_seconds"]
        positive_labels: list[str] = [
            self.machine_to_display_mapping[x]
            for x in meta_row["positive_labels"].split(",")
        ]

        download_paths: list[Path] = [
            Path(
                self.download_dir / label / Path(f"{ytid}_{start_time}-{end_time}.wav")
            )
            for label in positive_labels
        ]
        # print(f"Downloading {ytid} from {start_time} to {end_time} to {download_paths}")
        for paths in download_paths:
            paths.parent.mkdir(exist_ok=True)

        ret_pair: tuple[int, Optional[Exception]] = download_audio_section(
            ytid, start_time, end_time, download_paths, "wav", True
        )

        assert (
            ret_pair is not None
        ), f"Error downloading {ytid} from {start_time} to {end_time}"

        # print(f"{ret_pair=}")
        # while we still keep getting bot sniped, try again
        while ret_pair[
            0
        ] == 1 and "Sign in to confirm you\u2019re not a bot. This helps protect our community. Learn more" in str(
            ret_pair[1]
        ):
            print(
                f"Job number {job_id} got bot sniped. Retrying download for ytid={ytid} after sleeping for {self.sleep_amount} seconds"
            )
            time.sleep(self.sleep_amount)
            ret_pair = download_audio_section(
                ytid, start_time, end_time, download_paths, "wav", True
            )

        assert (
            ret_pair is not None
        ), f"Error downloading {ytid} from {start_time} to {end_time}"

        with lock:
            if ret_pair[0] == 1 or ret_pair[1] is not None:
                # print(f"{dict['Errors']=}")
                # print(f"{dict['Error_Ids']=}")
                dict["Errors"] += [str(ret_pair[1])]
                dict["Error_Ids"] += [ytid]
            elif ret_pair[0] == 0:
                # print(f"{dict['Downloaded_Ids']}")
                dict["Downloaded_Ids"] += [ytid]

    def init_multipart_download(self):

        with mp.Manager() as manager:

            # Split URLs across workers

            metadata_list: list[pd.DataFrame] = self.split_ids(
                self.filtered_split_df, self.num_jobs
            )
            print(f"{metadata_list=}")

            worker_processes: list[Worker] = []

            for job_id, meta_slice in enumerate(metadata_list):
                dict_prox: DictProxy = manager.dict()
                dict_prox["Errors"] = []
                dict_prox["Error_Ids"] = []
                dict_prox["Downloaded_Ids"] = []
                lock: mp.Lock = manager.Lock()
                process: mp.Process = mp.Process(
                    target=self.worker, args=(meta_slice, dict_prox, job_id, lock)
                )

                worker_processes.append(Worker(dict_prox, lock, job_id, process))

            logger_process: mp.Process = mp.Process( target=self.logger, args=(worker_processes, self.total_num_files))
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
