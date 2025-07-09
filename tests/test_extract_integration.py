# tests/test_extract_integration.py
"""
Integration test for the whole extract pipeline
------------------------------------------------
The flow should:
  1. "Download" a ZIP (stubbed – no real HTTP)
  2. Unpack             → student-mat.csv / student-por.csv
  3. Clean + engineer    → students_clean.parquet
  4. Split               → train.parquet + test.parquet
We assert that the final train.parquet exists where we expect.
"""

import io
import urllib.request
import zipfile
from importlib import util
from pathlib import Path

# ── 1️⃣  Dynamically load flows/00_extract_flow.py ────────────────
project_root = Path(__file__).resolve().parents[1]
mod_path = project_root / "flows" / "_00_extract_flow.py"

spec = util.spec_from_file_location("extract_flow", mod_path)
extract_flow = util.module_from_spec(spec)  # type: ignore
assert spec.loader is not None
spec.loader.exec_module(extract_flow)  # makes flow & tasks visible

# alias for convenience
flow = extract_flow.extract_flow

# ── 2️⃣  Build an in-memory ZIP with *both* required CSVs ────────────────
csv = (
    "school;sex;age;G1;G2;G3\n"  # minimal schema – just enough for treat_data()
    "GP;F;18;10;11;12\n"
)

zip_bytes = io.BytesIO()
with zipfile.ZipFile(zip_bytes, mode="w") as zf:
    zf.writestr("student-mat.csv", csv)
    zf.writestr("student-por.csv", csv)
zip_bytes.seek(0)


# ── 3️⃣  Stub urllib so *no* external call is made ───────────────────────
class _DummyResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def _fake_urlopen(url, timeout=60):  # new BytesIO each time
    return _DummyResp(zip_bytes.getvalue())


# ── 4️⃣  The actual integration test ─────────────────────────────────────
def test_extract_flow_end_to_end(tmp_path, monkeypatch):
    """Full flow should create train.parquet inside YYYY_MM_DD folder."""

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    # Run the *real* Prefect flow (sync, in-process)
    flow(
        url="http://dummy.test/zip",  # value is irrelevant after patching
        base_dir=tmp_path.as_posix(),
    )

    # One timestamped directory should now exist under tmp_path
    subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(subdirs) == 1, "Flow must create exactly one data folder"
    ts_dir = subdirs[0]

    # Final artefacts expected from split_train_test()
    assert (ts_dir / "train.parquet").is_file(), "train.parquet missing"
    assert (ts_dir / "students_clean.parquet").is_file(), "cleaned data missing"
