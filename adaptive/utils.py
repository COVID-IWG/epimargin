import os.path
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable, Union
from urllib.request import urlopen

import pandas as pd
from rich.progress import (BarColumn, DownloadColumn, Progress, TaskID,
                           TextColumn, TimeRemainingColumn,
                           TransferSpeedColumn)
from urlpath import URL

days  = 1
weeks = 7

def cwd() -> Path:
    argv0 = sys.argv[0]
    if argv0.endswith("ipython") or "venv" in argv0:
        return Path(".").resolve()
    return Path(argv0).resolve().parent
        
def fmt_params(delim=": ", **kwargs) -> str:
    return ", ".join(f"{k.replace('_', ' ')}{delim}{v}" for (k, v) in kwargs.items())

def assume_missing_0(df: pd.DataFrame, col: str):
    return df[col] if col in df.columns else 0


# download utils 
progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    DownloadColumn(),
    "•",
    TransferSpeedColumn(),
    "•",
    TimeRemainingColumn(),
)

def copy_url(task_id: TaskID, url: URL, path: Path) -> None:
    """Copy data from a url to a local file."""
    response = urlopen(str(url))
    # This will break if the response doesn't contain content length
    progress.update(task_id, total=int(response.info()["Content-length"]))
    with path.open('wb') as dest_file:
        progress.start_task(task_id)
        for data in iter(partial(response.read, 32768), b""):
            dest_file.write(data)
            progress.update(task_id, advance=len(data))


def download(src: str, dst: Path, filenames: Iterable[str]):
    """Download filenames from a base URL (src) to a local directory (dst)."""
    src = URL(src)
    with progress, ThreadPoolExecutor(max_workers=4) as pool:
        for filename in filenames:
            task_id = progress.add_task("download", filename=filename, start=False)
            pool.submit(copy_url, task_id, src/filename, dst/filename)
