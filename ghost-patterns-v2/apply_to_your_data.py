"""
Run the full validated pipeline on your own dataset.

    python apply_to_your_data.py mydata.csv
    python apply_to_your_data.py mushrooms.csv --target Mushroom_quality

Kept for backward compatibility. It is the same as

    ghostpatterns analyse mydata.csv [options]

Results go to results/<dataset name>/. Run with --help for every option.
"""

from __future__ import annotations

import sys

from ghostpatterns.cli import run


def main(argv: list[str] | None = None) -> str:
    return run(["analyse", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main()
