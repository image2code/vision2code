from __future__ import annotations

import argparse
from pathlib import Path

from vision2code.utils.io import read_csv, write_csv


def summarize_tool_use(eval_csv: Path, output_csv: Path):
    rows = read_csv(eval_csv)
    out = [{"input_csv": eval_csv.as_posix(), "rows": len(rows)}]
    write_csv(output_csv, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a tool-use evaluation CSV.")
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for row in summarize_tool_use(args.eval_csv, args.output):
        print(row)


if __name__ == "__main__":
    main()
