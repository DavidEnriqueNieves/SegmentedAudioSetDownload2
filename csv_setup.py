from pathlib import Path
import pandas as pd
import time


class CsvDownloader:
    """
    Class for downloading the CSVs required for the AudioSet dataset
    """

    split_names: list[str] = [
        "eval_segments",
        "balanced_train_segments",
        "unbalanced_train_segments",
    ]

    # all obtained by downloading the CSVs and counting the number of rows, minus the header row
    # The number of segments in each split of the dataset
    # used to double check that the number of segments in the cached metadata CSV is correct
    # NOTE: these numbers do NOT include the header row of the CSV
    split_quantities: dict[str, int] = {
        "eval_segments": 20373,
        "balanced_train_segments": 22162,
        "unbalanced_train_segments": 2041791,
    }

    # obtained by downloading a fresh copy of the CSV and counting the number of rows, minus the header row
    # NOTE: this number does NOT include the header row of the CSV
    class_to_label_csv_length: int = 527

    def __init__(self, split_name: str, cache_dir: Path):

        self.split_name: str = split_name
        self.segments_file: Path = cache_dir /  Path(f"{self.split_name}.csv")
        self.class_labels_file: str = cache_dir / Path("class_labels_indices.csv")

        # define URL to get metadata from
        self.segment_csv_url: str = (
            f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{self.split_name}.csv"
        )
        # refers to the "class labels indices" or "labels meta" CSV that one
        # apparently needs to translate the labels from the metadata into human
        # readable labels"
        self.class_label_idxs_url: str = (
            f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
        )

        self.cache_dir: Path = cache_dir

    def load_segment_csv_url(self, n_splits : int, split_idx : int, dataset_nrows : int) -> pd.DataFrame:
        """Downloads the segment URL based off the 'download_type' and foregoes the download if the metadata is present in the cache directory"""
        print("Loading metadata CSV...")
        if not self.segments_file.exists():
            # Load the metadata
            print("Cached csv file not detected.")
            print(f"Downloading from {self.segment_csv_url}")

            start : float = time.time()
            self.metadata: pd.DataFrame = pd.read_csv(
                self.segment_csv_url,
                sep=", ",
                skiprows=3,
                header=None,
                names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
                engine="python",
            )
            end : float = time.time()
            print(f"Downloaded {self.split_name} segment CSV in {end-start} seconds")

            # this sep is pretty important since the others depend on it
            # Why "|", instead of ","?
            # it's because of there being a quoted comma separated string inside the CSV containing all the labels
            # frustratingly enough, the separator can only be a one-character string
            self.metadata.to_csv(str(self.segments_file), sep="|")
            print(f"Saved to {self.segments_file}")

        else:

            start : float = time.time()
            # we only need to load the split part of the CSV
            # means loading only rows from 1 + split_idx * dataset_nrows to 1 + (split_idx + 1) * dataset_nrows

            start_idx : int = 3 + split_idx * int(dataset_nrows / n_splits)
            self.metadata: pd.DataFrame = pd.read_csv(
                str(self.segments_file),
                sep="|",
                skiprows= start_idx,
                nrows=int(dataset_nrows / n_splits),
                header=None,
                names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
                engine="python",
            )
            end : float = time.time()
            print(f"Loaded {self.split_name} segment CSV from cache in {end-start} seconds")

            print(f"Cached csv file detected at {self.segments_file}")

        # even though the CSV metadata file might be cached, we check to make sure
        # the number of segments in the CSV matches the expected number of
        # segments
        # assert (
        #     self.metadata.shape[0] == self.split_quantities[self.split_name]
        # ), f"Number of segments in the cached metadata CSV does not match the expected number of segments for the split {self.split_name}"

        self.metadata["positive_labels"] = self.metadata["positive_labels"].apply(
            lambda x: x.replace('"', "")
        )
        self.metadata = self.metadata.reset_index(drop=True)

        return self.metadata

    def load_class_mapping_csv(self) -> pd.DataFrame:
        """
        Loads the CSV file which maps from the machine labels to the human-readable labels
        """

        print("Loading label map CSV...")
        if not self.class_labels_file.exists():
            start : float = time.time()
            print("Cached label meta csv file not detected.")
            print(f"Downloading from {self.class_label_idxs_url}")
            self.label_meta_df = pd.read_csv(
                self.class_label_idxs_url,
                sep=",",
            )
            end : float = time.time()
            print(f"Downloaded class mapping CSV in {end-start} seconds")

            self.label_meta_df.to_csv(self.class_labels_file)
            print(f"Saved to path {self.class_labels_file}")
        else:
            print(
                f"Cached label meta csv file detected at path {self.class_labels_file}"
            )
            start : float = time.time()
            self.label_meta_df = pd.read_csv(
                self.class_labels_file,
                sep=",",
            )
            end : float = time.time()
            print(f"Loaded class mapping CSV from cache in {end-start} seconds")

        # even though the CSV metadata file might be cached, we check to make sure
        # the number of rows in the CSV matches the expected number of rows
        # assert (
        #     self.label_meta_df.shape[0] == self.class_to_label_csv_length
        # ), f"Number of rows in the cached label meta CSV does not match the expected number of rows"

        self.display_to_machine_mapping = dict(
            zip(self.label_meta_df["display_name"], self.label_meta_df["mid"])
        )
        self.machine_to_display_mapping = dict(
            zip(self.label_meta_df["mid"], self.label_meta_df["display_name"])
        )
        return self.label_meta_df
