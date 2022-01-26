import math
from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import cast

from text_selection.selection import SelectionMode
from text_selection_core.selection.fifo import select_fifo
from text_selection_core.selection.select_greedy import select_greedy

from text_selection_app.argparse_helper import (ConvertToOrderedSetAction,
                                                parse_existing_directory,
                                                parse_non_empty_or_whitespace,
                                                parse_non_negative_float)
from text_selection_app.helper import get_datasets
from text_selection_app.io import (get_data_n_grams_path,
                                   get_data_weights_path, get_dataset_path,
                                   load_data_n_grams, load_data_weights,
                                   load_dataset, save_dataset)


def get_fifo_selection_parser(parser: ArgumentParser):
  parser.description = f"Select Id's by FIFO principle."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("from_subsets", type=parse_non_empty_or_whitespace, nargs="+", metavar="from-subsets",
                      help="from subset", action=ConvertToOrderedSetAction)
  parser.add_argument("to_subset", type=parse_non_empty_or_whitespace, metavar="to-subset",
                      help="to subset")
  parser.add_argument("--weights", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="weights name", default="weights")
  parser.add_argument("--limit", type=parse_non_negative_float, metavar="FLOAT",
                      help="weights limit", default=math.inf)
  parser.add_argument("-i", "--limit-include-already-selected", action="store_true",
                      help="include already selected Id's for limit")
  parser.add_argument("-p", "--limit-percent", action="store_true",
                      help="limit is percentual; in this case it needs to be in interval (0, 100]")
  return select_fifo_from_ns


def select_fifo_from_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")
    weights_path = get_data_weights_path(data_folder, ns.weights)
    if not weights_path.exists():
      logger.error(
        f"Weights were not found! Skipping...")
      continue

    dataset = load_dataset(dataset_path)
    weights = load_data_weights(weights_path)

    error, changed_anything = select_fifo(dataset, ns.from_subsets, ns.to_subset, weights,
                                          ns.limit, ns.limit_include_already_selected, ns.limit_percent)

    success = error is None

    if not success:
      logger.error(f"{error.default_message}")
      logger.info("Skipped.")
      assert not changed_anything
    else:
      if changed_anything:
        save_dataset(get_dataset_path(data_folder), dataset)
      else:
        logger.info("Didn't changed anything!")


def get_greedy_selection_parser(parser: ArgumentParser):
  parser.description = f"Select Id's by greedy principle."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("from_subsets", type=parse_non_empty_or_whitespace, nargs="+", metavar="from-subsets",
                      help="from subset", action=ConvertToOrderedSetAction)
  parser.add_argument("to_subset", type=parse_non_empty_or_whitespace, metavar="to-subset",
                      help="to subset")
  parser.add_argument("--n-grams", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="n-grams name", default="n-grams")
  parser.add_argument("--include-selected", action="store_true",
                      help="consider already selected for the selection")
  parser.add_argument("--weights", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="weights name", default="weights")
  parser.add_argument("--limit", type=parse_non_negative_float, metavar="FLOAT",
                      help="weights limit", default=math.inf)
  parser.add_argument("-i", "--limit-include-already-selected", action="store_true",
                      help="include already selected Id's for limit")
  parser.add_argument("-p", "--limit-percent", action="store_true",
                      help="limit is percentual; in this case it needs to be in interval (0, 100]")
  return greedy_selection_ns


def greedy_selection_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    weights_path = get_data_weights_path(data_folder, ns.weights)
    if not weights_path.exists():
      logger.error(
        f"Weights were not found! Skipping...")
      continue

    n_grams_path = get_data_n_grams_path(data_folder, ns.n_grams)
    if not n_grams_path.exists():
      logger.error(
        f"N-Grams were not found! Skipping...")
      continue

    dataset = load_dataset(dataset_path)
    weights = load_data_weights(weights_path)
    n_grams = load_data_n_grams(n_grams_path)

    error, changed_anything = select_greedy(dataset, ns.from_subsets, ns.to_subset, n_grams, weights,
                                            ns.limit, ns.limit_include_already_selected, ns.limit_percent, ns.include_selected, SelectionMode.FIRST)

    success = error is None

    if not success:
      logger.error(f"{error.default_message}")
      logger.info("Skipped.")
      assert not changed_anything
    else:
      if changed_anything:
        save_dataset(get_dataset_path(data_folder), dataset)
      else:
        logger.info("Didn't changed anything!")