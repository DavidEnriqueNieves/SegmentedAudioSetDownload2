from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.managers import DictProxy
from typing import Dict

@dataclass
class Worker:
    """
    Simple class for bundling a dictionary proxy, a lock object, and a job ID.
    """
    dict_proxy: DictProxy
    lock: mp.Lock
    job_id: int
    process : mp.Process
