from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import cast

import pandas as pd
from ordered_set import OrderedSet

from text_selection_app.argparse_helper import (ConvertToOrderedSetAction, parse_existing_directory,
                                                parse_non_empty, parse_non_empty_or_whitespace)
from text_selection_app.helper import get_datasets
from text_selection_app.io_handling import get_data_weights_path, load_data_weights, load_dataset
from text_selection_core.statistics import generate_statistics


def get_statistics_generation_parser(parser: ArgumentParser):
  parser.description = "This command creates statistics."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("--weights", type=parse_non_empty_or_whitespace, nargs="*", metavar="NAME",
                      help="name of the weights", default=[], action=ConvertToOrderedSetAction)
  parser.add_argument("file", type=parse_non_empty_or_whitespace,
                      help="name of the file")
  parser.add_argument("--lsep", type=parse_non_empty, default="\n",
                      help="line separator")
  parser.add_argument("--ssep", type=str, default="",
                      help="symbol separator")
  parser.add_argument("-o", "--overwrite", action="store_true",
                      help="overwrite")
  return statistics_generation_ns


def statistics_generation_ns(ns: Namespace) -> None:
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    weights = []
    for weights_name in cast(OrderedSet[str], ns.weights):
      weights_path = get_data_weights_path(data_folder, weights_name)
      if not weights_path.is_file():
        logger.error(
          f"Weights '{weights_name}' were not found! Ignoring...")
        continue
      current_weights = load_data_weights(weights_path)
      weights.append((weights_name, current_weights))

    symbols_path = data_folder / cast(str, ns.file)
    if not symbols_path.exists():
      logger.error(
        "File was not found! Skipping...")
      continue
    lines = symbols_path.read_text(ns.encoding).split(ns.lsep)

    dataset = load_dataset(dataset_path)
    logger.debug("Generating statistics...")
    dfs = generate_statistics(dataset, lines, ns.ssep, weights)
    stats_path = root_folder / "statistics.csv"
    header_indicator = "###"
    with open(stats_path, mode="w") as f:
      f.write(f"{header_indicator} Statistics {header_indicator}\n\n")

    for df_name, df in dfs:
      logger.debug(f"Saving {df_name}...")

      with open(stats_path, mode="a") as f:
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
