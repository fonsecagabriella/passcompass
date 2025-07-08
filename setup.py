# setup.py
from setuptools import find_packages, setup

setup(
    name="passcompass_utils",
    version="0.1.0",
    package_dir={"": "src"},  # <-- tells setuptools where packages live
    packages=find_packages(where="src"),
    python_requires=">=3.8",
)
