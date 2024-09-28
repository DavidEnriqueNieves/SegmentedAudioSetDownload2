from typing import Optional
from pathlib import Path
import yt_dlp
import shutil
import time
import traceback

"""
Script for definining mostly static functions for downloading youtube videos using youtube-dlp
"""

def get_current_time_ms():
    return time.time() * 1000


# https://stackoverflow.com/questions/71326109/how-to-hide-error-message-from-youtube-dl-yt-dlp
class loggerOutputs:
    def error(msg):
        # print("Captured Error: "+msg)
        # lel, we want this thing to SHUT UP
        1
    def warning(msg):
        # print("Captured Warning: "+msg)
        2
    def debug(msg):
        # print("Captured Log: "+msg)
        3

def download_audio_section(
    ytid: str,
    start_time: int,
    end_time: int,
    dwnld_paths: list[Path],
    codec_type: str = "wav",
    quiet: bool = True,
) -> tuple[int, Optional[Exception]]:

    url: str = f"https://www.youtube.com/watch?v={ytid}"
    ydl_opts = {
        "quiet": quiet,
        "no_warnings": quiet,  # Suppress warnings if quiet is True
        "format": "bestaudio/best",
        # "logger" : loggerOutputs,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": codec_type,
                "preferredquality": "192",
            }
        ],
        "outtmpl": str(dwnld_paths[0].with_suffix(".%(ext)s")),
        "download_ranges": lambda info, _: [
            {
                "start_time": start_time,
                "end_time": end_time,
            }
        ],
    }

    ex: Exception = None

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            retcode: int = ydl.download([url])

            if len(dwnld_paths) > 1:
                for path in dwnld_paths[1:]:
                    shutil.copy(
                        dwnld_paths[0].with_suffix(f".{codec_type}"),
                        path.with_suffix(f".{codec_type}"),
                    )
            return (retcode, None)

        except Exception as e:
            # raise e
            ex = e
            # traceback.print_exc()
            # NOTE: the lack of exception need not imply the video downloaded successfully
            # print("No errors")
            return (1, e)