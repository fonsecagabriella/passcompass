"""
flows/extract_flow.py
Usage
-----
python flows/extract_flow.py                         # one-off local run
python flows/extract_flow.py --base-dir /tmp/data    # custom location
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse, io, zipfile, urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from prefect import flow, task, get_run_logger

# ────────────────────────────────────────────────────────────────────────────
# Defaults ───────────────────────────────────────────────────────────────────
UCI_URL = (  # one public ZIP – no auth needed
    "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
)
BASE_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "passcompass"
).as_posix()  # one level above “pipeline”

# ────────────────────────────────────────────────────────────────────────────
# TASKS ──────────────────────────────────────────────────────────────────────
@task(retries=4, retry_delay_seconds=30, log_prints=True)
def download_and_extract(url: str, base_dir: str) -> Path:
    """Download the zip and extract everything into a timestamped folder."""
    ts_dir = Path(base_dir) / datetime.utcnow().strftime("%Y_%m_%d")
    ts_dir.mkdir(parents=True, exist_ok=True)

    logger = get_run_logger()
    logger.info(f"Fetching {url}")

    # stream → unzip in memory so we don't leave zip files on disk
    with urllib.request.urlopen(url, timeout=60) as resp:
        with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            zf.extractall(ts_dir)

    # also unfold *any* nested zip files they include
    for f in ts_dir.glob("*.zip"):
        logger.info(f"Unfolding nested archive {f.name}")
        with zipfile.ZipFile(f) as zf:
            zf.extractall(ts_dir)
        f.unlink()

    return ts_dir  # directory path


@task(log_prints=True)
def treat_data(dir_path: Path) -> Path:
    """Combine math & Portuguese datasets and engineer the target."""
    math_df = pd.read_csv(dir_path / "student-mat.csv", sep=";")
    por_df  = pd.read_csv(dir_path / "student-por.csv", sep=";")

    math_df["course"] = "math"
    por_df["course"]  = "por"

    students = (
        pd.concat([math_df, por_df], ignore_index=True)
        .assign(pass_=lambda df: (df["G3"] >= 10).astype(int))
        .drop(columns=["G1", "G2", "G3"])
        .rename(columns={"pass_": "pass"})
    )

    out_path = dir_path / "students_clean.parquet"
    students.to_parquet(out_path, index=False)

    get_run_logger().info(f"Treated data saved to {out_path}")
    return out_path


@task(log_prints=True)
def split_train_test(data_path: Path, test_size: float = 0.2, seed: int = 42):
    df = pd.read_parquet(data_path)
    train, test = train_test_split(df, test_size=test_size, random_state=seed)

    train_path = data_path.with_name("train.parquet")
    test_path  = data_path.with_name("test.parquet")
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path,  index=False)
    return train_path, test_path


@task(log_prints=True)
def basic_stats(train_path: Path):
    df = pd.read_parquet(train_path)
    pass_rate = df["pass"].mean()
    fail_rate = 1 - pass_rate
    get_run_logger().info(
        {
            "rows": len(df),
            "pass_rate": f"{pass_rate:.2%}",
            "fail_rate": f"{fail_rate:.2%}",
        }
    )


# ────────────────────────────────────────────────────────────────────────────
# FLOW ───────────────────────────────────────────────────────────────────────
#@flow(name="extract_flow", log_prints=True, tags=["project:passcompass", "stage:dev", "type:extract"])
@flow(name="extract_flow", log_prints=True)
def extract_flow(url: str = UCI_URL, base_dir: str = BASE_DIR):


    data_dir      = download_and_extract(url, base_dir)     # 1
    cleaned_path  = treat_data(data_dir)                    # 2
    train_path, _ = split_train_test(cleaned_path)          # 3
    basic_stats(train_path)                                 # 4


# ────────────────────────────────────────────────────────────────────────────
# CLI one-off run  ----------------------------------------------------------
if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--url", default=UCI_URL)
    cli.add_argument("--base-dir", default=BASE_DIR)
    args = cli.parse_args()

    # one-off run (no schedule) so you can test locally:
    extract_flow(args.url, args.base_dir)

    # comment-in when you’re ready to register a 6-month deployment
    # extract_flow.serve(
    #     name     ="extract-every-6-months",
    #     cron     ="0 0 1 */6 *",        # 1 Jan & 1 Jul 00:00
    #     tags     ={"project": "mlops-zoomcamp"},
    #     timezone ="Europe/Amsterdam",
    # )
