#!/usr/bin/env python
"""
Start an MLflow tracking server using values from conf/mlflow_local.yaml.
Compatible with MLflow < 2.24 (no --config-file flag).
"""
import subprocess, pathlib, yaml, sys

CONF = pathlib.Path(__file__).parents[1] / "conf" / "mlflow_local.yaml"

try:
    cfg = yaml.safe_load(CONF.read_text())
except FileNotFoundError:
    sys.exit("Config file not found. Expected at conf/mlflow_local.yaml")

cmd = [
    "mlflow", "server",
    "--backend-store-uri", cfg["backend_store_uri"],
    "--default-artifact-root", cfg["default_artifact_root"],
    "--host", str(cfg.get("host", "127.0.0.1")),
    "--port", str(cfg.get("port", 5000)),
]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
print(f"MLflow server started at {cfg['host']}:{cfg['port']}")