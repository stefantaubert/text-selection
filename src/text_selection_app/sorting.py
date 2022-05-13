import math
from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import cast

from text_selection.selection import SelectionMode
from text_selection_app.argparse_helper import (ConvertToOrderedSetAction, parse_existing_directory,
                                                parse_non_empty_or_whitespace,
                                                parse_non_negative_float)
from text_selection_app.default_args import add_directory_argument
from text_selection_app.helper import get_datasets
from text_selection_app.io_handling import (get_data_weights_path, get_dataset_path,
                                            try_load_data_weights, try_load_dataset,
                                            try_save_dataset)
from text_selection_core.common import (SelectionDefaultParameters, SortingDefaultParameters,
                                        WeightSelectionParameters)
from text_selection_core.selection.greedy_selection import GreedySelectionParameters, select_greedy
from text_selection_core.sorting.fifo_sorting import sort_fifo
from text_selection_core.sorting.reverse_sorting import sort_reverse


def get_fifo_sorting_parser(parser: ArgumentParser):
  parser.description = "Sort lines by FIFO principle."
  add_directory_argument(parser)
  parser.add_argument("subsets", type=parse_non_empty_or_whitespace, nargs="+", metavar="subsets",
                      help="subsets", action=ConvertToOrderedSetAction)
  return sort_fifo_from_ns


def sort_fifo_from_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    dataset = try_load_dataset(dataset_path, logger)
    if dataset is None:
      logger.info("Skipped!")
      continue

    default_params = SortingDefaultParameters(dataset, ns.subsets)
    error, changed_anything = sort_fifo(default_params)

    success = error is None

    if not success:
      print(error)
      logger.error(f"{error.default_message}")
      logger.info("Skipped.")
      assert not changed_anything
    else:
      if changed_anything:
        try_save_dataset(get_dataset_path(data_folder), dataset, logger)
      else:
        logger.info("Didn't changed anything!")


def get_reverse_sorting_parser(parser: ArgumentParser):
  parser.description = "Reverse sorting."
  add_directory_argument(parser)
  parser.add_argument("subsets", type=parse_non_empty_or_whitespace, nargs="+", metavar="subsets",
                      help="subsets", action=ConvertToOrderedSetAction)
  return sort_reverse_from_ns


def sort_reverse_from_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    dataset = try_load_dataset(dataset_path, logger)
    if dataset is None:
      logger.info("Skipped!")
      continue

    default_params = SortingDefaultParameters(dataset, ns.subsets)
    error, changed_anything = sort_reverse(default_params)

    success = error is None

    if not success:
      print(error)
      logger.error(f"{error.default_message}")
      logger.info("Skipped.")
      assert not changed_anything
    else:
      if changed_anything:
        try_save_dataset(get_dataset_path(data_folder), dataset, logger)
      else:
        logger.info("Didn't changed anything!")
