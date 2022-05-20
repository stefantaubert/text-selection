from argparse import ArgumentParser, Namespace
from logging import Logger
from pathlib import Path
from typing import cast

import pandas as pd
from ordered_set import OrderedSet

from text_selection_cli.argparse_helper import (ConvertToOrderedSetAction, parse_existing_file,
                                                parse_path)
from text_selection_cli.default_args import (add_file_arguments, add_dataset_argument)
from text_selection_cli.io_handling import (try_load_data_weights, try_load_dataset,
                                            try_load_file)
from text_selection_core.globals import ExecutionResult
from text_selection_core.statistics import generate_statistics


def get_statistics_generation_parser(parser: ArgumentParser):
  parser.description = "This command creates statistics as csv-file."
  add_dataset_argument(parser)
  parser.add_argument("path", type=parse_path, metavar="OUTPUT-PATH", help="statistics output path (csv)")
  parser.add_argument("--weights", type=parse_existing_file, nargs="*", metavar="PATH",
                      help="path to weights", default=OrderedSet(), action=ConvertToOrderedSetAction)
  add_file_arguments(parser, True)
  return statistics_generation_ns


def statistics_generation_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.dataset, logger)
  if dataset is None:
    return False, False

  weights = []
  for weights_path in cast(OrderedSet[Path], ns.weights):
    current_weights = try_load_data_weights(ns.weights, logger)
    if current_weights is None:
      return False, False

    weights.append((weights_path.stem, current_weights))

  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if lines is None:
    return False, False

  logger.debug("Generating statistics...")
  dfs = generate_statistics(dataset, lines, ns.sep, weights, flogger)

  stats_path = cast(Path, ns.path)
  header_indicator = "###"
  with open(stats_path, mode="w", encoding="UTF-8") as f:
    f.write(f"{header_indicator} Statistics {header_indicator}\n\n")

  for df_name, df in dfs:
    logger.debug(f"Saving {df_name}...")

    with open(stats_path, mode="a", encoding="UTF-8") as f:
      header = f"{header_indicator} {df_name} {header_indicator}"
      f.write(f"{header}\n\n")
      df.to_csv(f, sep=";", index=False)
      f.write("\n")

      with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.precision', 7,
        'display.width', None,
      ):
        print(header)
        print(df)
        print()
  return True, None
