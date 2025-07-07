# tests/test_download.py
from pathlib import Path
from src.flows import download_data   # your Prefect flow / task

def test_download_creates_file(tmp_path):
    target = tmp_path/"sample.csv"
    download_data(path=str(target), sample=True)  # sample=True → tiny file
    assert target.exists()
    assert target.stat().st_size > 0        # not empty
