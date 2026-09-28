"""
Command-line interface.

    ghostpatterns analyse data.csv --target label
    ghostpatterns leakage data.csv
    ghostpatterns benchmark --quick
    ghostpatterns figures
    ghostpatterns app

Also available as `python -m ghostpatterns ...`.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from . import __version__
from .analyse import Options


class Tee:
    """Write console output to the terminal and a log file at the same time."""

    def __init__(self, path: str):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.file.write(text)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        sys.stdout = self.stdout
        self.file.close()


def _add_analyse(sub) -> None:
    d = Options()
    p = sub.add_parser(
        "analyse", aliases=["analyze"],
        help="find patterns and keep only the ones that beat noise",
        description="Run leakage checks, calibrated rule mining, column "
                    "attribution, clustering and anomaly tests on a CSV.")
    p.add_argument("path", help="CSV file to analyse")
    p.add_argument("--target", default=None,
                   help="only test rules that predict this column; "
                        "far sharper results")
    p.add_argument("--outdir", default=None,
                   help="output folder (default: results/<dataset name>)")
    p.add_argument("--n-null", type=int, default=d.n_null,
                   help=f"null replicates (default {d.n_null})")
    p.add_argument("--alpha", type=float, default=d.alpha)
    p.add_argument("--min-support", type=float, default=d.min_support)
    p.add_argument("--min-confidence", type=float, default=d.min_confidence)
    p.add_argument("--min-lift", type=float, default=d.min_lift)
    p.add_argument("--max-len", type=int, default=d.max_len)
    p.add_argument("--max-cardinality", type=int, default=d.max_cardinality)
    p.add_argument("--drop-leaking", action="store_true",
                   help="automatically drop columns flagged as hard leakage")
    p.add_argument("--no-attribution", action="store_true",
                   help="skip column attribution (the slowest step)")
    p.add_argument("--open", action="store_true",
                   help="open the HTML report when done")
    p.add_argument("--seed", type=int, default=d.seed)
    p.set_defaults(func=cmd_analyse)


def cmd_analyse(args) -> str:
    from .analyse import analyse, save

    name = os.path.splitext(os.path.basename(args.path))[0]
    out = args.outdir or os.path.join("results", name)
    os.makedirs(out, exist_ok=True)
    opts = Options(target=args.target, min_support=args.min_support,
                   min_confidence=args.min_confidence, min_lift=args.min_lift,
                   max_len=args.max_len, max_cardinality=args.max_cardinality,
                   n_null=args.n_null, alpha=args.alpha, seed=args.seed,
                   drop_leaking=args.drop_leaking,
                   attribution=not args.no_attribution)

    tee = Tee(os.path.join(out, "log.txt"))
    sys.stdout = tee
    try:
        df = pd.read_csv(args.path)
        if args.target and args.target not in df.columns:
            raise SystemExit(f"--target {args.target!r} is not a column. "
                             f"Columns: {list(df.columns)}")
        if not args.target:
            print("Tip: pass --target <column> to test only rules that "
                  "predict one column.")
        result = analyse(df, opts, name=name)
        paths = save(result, out)
        print(f"\nAll results saved in {out}{os.sep}")
        if "report.html" in paths:
            print(f"Open the report: {os.path.abspath(paths['report.html'])}")
    finally:
        tee.close()

    if args.open and "report.html" in paths:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(paths["report.html"]))
    return out


def cmd_leakage(args) -> None:
    from .leakage import check_leakage
    report = check_leakage(pd.read_csv(args.path), verbose=True)
    if args.csv:
        report.to_frame().to_csv(args.csv, index=False)
        print(f"\nSaved {args.csv}")
    sys.exit(0 if report.clean else 2)


def cmd_benchmark(args) -> None:
    from .benchmark import full_benchmark
    full_benchmark(quick=args.quick, verbose=True)


def cmd_figures(args) -> None:
    from .viz import generate_all
    generate_all(outdir=args.outdir, quick=not args.full)


def cmd_app(args) -> None:
    import subprocess
    app = os.path.join(os.path.dirname(__file__), "app.py")
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit("The web app needs streamlit: "
                         "pip install \"ghostpatterns[app]\"")
    theme = ["--theme.base", "dark", "--theme.primaryColor", "#38bdf8",
             "--theme.backgroundColor", "#0b1120",
             "--theme.secondaryBackgroundColor", "#111827",
             "--theme.textColor", "#e2e8f0"]
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", app,
                              *theme]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ghostpatterns",
        description="Tell real patterns from ghosts: every finding is "
                    "calibrated against what structureless noise produces.")
    parser.add_argument("--version", action="version",
                        version=f"ghostpatterns {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_analyse(sub)

    p = sub.add_parser("leakage", help="check a CSV for self-inflicted "
                                       "structure; exit code 2 if found")
    p.add_argument("path")
    p.add_argument("--csv", default=None, help="also save findings to CSV")
    p.set_defaults(func=cmd_leakage)

    p = sub.add_parser("benchmark", help="measure the validator's own false "
                                         "positive rate, power and FDR")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("figures", help="regenerate the explainer figures")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--full", action="store_true",
                   help="more replicates (slower)")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("app", help="launch the interactive web app")
    p.set_defaults(func=cmd_app)
    return parser


def run(argv: list[str] | None = None):
    """Parse `argv` and run the command; returns the command's result."""
    args = build_parser().parse_args(argv)
    return args.func(args)


def main(argv: list[str] | None = None) -> None:
    # Console-script entry point. Returning a value here would become the
    # process exit status, so the result of run() is deliberately dropped.
    run(argv)


if __name__ == "__main__":
    main()
