"""Command-line interface: ``malta-rock download`` and ``malta-rock run``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from MaltaRock.logger import LEVELS, setup_logging


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malta-rock",
        description="MALTA-ROCK: lithology classification experiments on MALTA-WLOGS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-ll", "--log-level", default="INFO", choices=LEVELS, dest="log_level")
    parser.add_argument("-d", "--data-path", default="./data", dest="data_path",
                        help="Root directory of the local MALTA-WLOGS copy.")
    parser.add_argument("--revision", default=None,
                        help="Dataset branch, tag or commit (pin a commit for reproducibility).")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download the bronze-feature dataset if needed.")
    download.add_argument("--force", action="store_true", help="Re-download even if cached.")

    run = commands.add_parser("run", help="Run an experiment from a YAML configuration.")
    run.add_argument("--model-config", required=True, dest="model_config")
    run.add_argument("--num-trials", type=int, default=None, dest="num_trials",
                     help="Optuna trials (default: 'trials' from the config).")
    run.add_argument("--cv-folds", type=int, default=3, dest="cv_folds",
                     help="Well-grouped CV folds used to score Optuna trials.")
    run.add_argument("--db-name", default=None, dest="db_name",
                     help="SQLite file to persist the Optuna study.")
    run.add_argument("--split-file", default=None, dest="split_file",
                     help="JSON with explicit train_wells/test_wells.")
    run.add_argument("--output", default="results", help="Directory for results and well split.")
    return parser


def cli(argv: Sequence[str] | None = None) -> None:
    """Entry point of the ``malta-rock`` command."""
    args = build_argparser().parse_args(argv)
    setup_logging(args.log_level)
    if args.command == "download":
        from MaltaRock.data.download import download_dataset

        download_dataset(args.data_path, force=args.force, revision=args.revision)
    elif args.command == "run":
        from MaltaRock.run import load_config, run_experiment

        run_experiment(
            load_config(args.model_config),
            data_path=args.data_path,
            revision=args.revision,
            n_trials=args.num_trials,
            cv_folds=args.cv_folds,
            storage=f"sqlite:///{args.db_name}" if args.db_name else None,
            output_dir=args.output,
            split_file=args.split_file,
        )


if __name__ == "__main__":
    cli()
