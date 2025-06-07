# setup.py
from setuptools import setup, find_packages

setup(
    name="passcompass_utils",
    version="0.1.0",
    package_dir={"": "src"},      # <-- tells setuptools where packages live
    packages=find_packages(where="src"),
    python_requires=">=3.8",
)
