from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import cast

import pandas as pd
from ordered_set import OrderedSet

from text_selection_cli.argparse_helper import (ConvertToOrderedSetAction, parse_existing_directory,
                                                parse_non_empty, parse_non_empty_or_whitespace)
from text_selection_cli.default_args import (add_directory_argument, add_file_arguments,
                                             parse_weights_name)
from text_selection_cli.helper import get_datasets
from text_selection_cli.io_handling import (get_data_weights_path, try_load_data_weights,
                                            try_load_dataset, try_load_file)
from text_selection_core.statistics import generate_statistics


def get_statistics_generation_parser(parser: ArgumentParser):
  parser.description = "This command creates statistics."
  add_directory_argument(parser)
  parser.add_argument("--weights", type=parse_weights_name, nargs="*", metavar="NAME",
                      help="name of the weights", default=[], action=ConvertToOrderedSetAction)
  add_file_arguments(parser, True)
  return statistics_generation_ns


def statistics_generation_ns(ns: Namespace) -> None:
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder, logger)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    weights = []
    for weights_name in cast(OrderedSet[str], ns.weights):
      weights_path = get_data_weights_path(data_folder, weights_name)
      current_weights = try_load_data_weights(weights_path, logger)
      if current_weights is None:
        logger.info("Skipped.")
        continue

      weights.append((weights_name, current_weights))

    dataset = try_load_dataset(dataset_path, logger)
    if dataset is None:
      logger.info("Skipped!")
      continue

    lines = try_load_file(data_folder / ns.file, ns.encoding, ns.lsep, logger)
    if lines is None:
      logger.info("Skipped!")
      continue

    logger.debug("Generating statistics...")
    dfs = generate_statistics(dataset, lines, ns.sep, weights)
    stats_path = root_folder / "statistics.csv"
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
