#!/usr/bin/env python3
"""Run the SCC semi-synthetic LR prioritization benchmark."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spagraph.analysis.semisynthetic_lr_benchmark import main


if __name__ == "__main__":
    main()
