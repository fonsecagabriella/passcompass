import io
import types
import urllib.request
import zipfile
from importlib import util
from pathlib import Path

# ──  Load the task module from its real path ────────────────────────────
project_root = Path(__file__).resolve().parents[1]  # adjust if tests/ nested deeper
mod_path = project_root / "flows" / "00_extract_flow.py"

spec = util.spec_from_file_location("extract_flow", mod_path)
extract_flow: types.ModuleType = util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(extract_flow)  # makes attributes available

download_and_extract = extract_flow.download_and_extract  # ← the task we need

# ── inject a dummy logger so Prefect isn't required ───────────────
import logging

extract_flow.get_run_logger = lambda: logging.getLogger("test")


# ──  Tests ────────────────────────────────────────────────────────────────
def test_download_and_extract_creates_files(tmp_path, monkeypatch):
    data = io.BytesIO()
    with zipfile.ZipFile(data, mode="w") as zf:
        zf.writestr("dummy.txt", "hello-world")
    data.seek(0)

    class _DummyResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    def _fake_urlopen(url, timeout=60):  # no real HTTP call
        return _DummyResp(data.getvalue())

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    out_dir: Path = download_and_extract.fn(  # call raw function, not Prefect Task
        url="http://irrelevant.test/student.zip",
        base_dir=tmp_path.as_posix(),
    )

    assert out_dir.is_dir()
    assert str(out_dir).startswith(str(tmp_path))
    assert (out_dir / "dummy.txt").is_file()
