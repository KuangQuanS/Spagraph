#!/usr/bin/env python3
"""Run the fully synthetic matched-control LR benchmark."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spagraph.analysis.synthetic_lr_v2_benchmark import main


if __name__ == "__main__":
    main()
