"""Build hooks; metadata and dependencies live in pyproject.toml."""

from pathlib import Path
import sys

from setuptools import setup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _build_support import BuildPy, SourceDistribution

setup(cmdclass={"build_py": BuildPy, "sdist": SourceDistribution})
