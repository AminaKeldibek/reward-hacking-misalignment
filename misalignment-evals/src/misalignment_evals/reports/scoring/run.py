"""Build the HTML report for one scoring run.

    python -m misalignment_evals.reports.scoring.run --config <report.yaml>

Writes the page at `output`, and its numbers beside it as .json.
"""
import argparse
from pathlib import Path

import yaml

from misalignment_evals.reports.scoring import html
from misalignment_evals.reports.scoring.analysis import build, write_json
from misalignment_evals.reports.scoring.config import ReportConfig


def load_config(path: "str | Path") -> ReportConfig:
    return ReportConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def run(config: ReportConfig) -> Path:
    data = build(config)
    write_json(data, config.output.with_suffix(".json"))
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(html.render(data))
    return config.output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(f"wrote {run(load_config(args.config))}")


if __name__ == "__main__":
    main()
