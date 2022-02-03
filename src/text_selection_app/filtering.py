from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import cast
from text_selection_core.common import SelectionDefaultParameters

from text_selection_core.filtering.duplicates_filter import filter_duplicates
from text_selection_core.filtering.regex_filter import filter_regex_pattern
from text_selection_app.argparse_helper import (ConvertToOrderedSetAction,
                                                parse_existing_directory,
                                                parse_non_empty_or_whitespace)
from text_selection_app.helper import get_datasets
from text_selection_app.io_handling import (get_data_symbols_path, get_dataset_path,
                                            load_data_symbols, load_dataset,
                                            save_dataset)


def get_duplicate_selection_parser(parser: ArgumentParser):
  parser.description = f"Select duplicate entries."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("from_subsets", type=parse_non_empty_or_whitespace, nargs="+", metavar="from-subsets",
                      help="from subset", action=ConvertToOrderedSetAction)
  parser.add_argument("to_subset", type=parse_non_empty_or_whitespace, metavar="to-subset",
                      help="to subset")
  return select_duplicates_ns


def select_duplicates_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    symbols_path = get_data_symbols_path(data_folder)
    if not symbols_path.exists():
      logger.error(
        f"Symbols were not found! Skipping...")
      continue

    dataset = load_dataset(dataset_path)
    symbols = load_data_symbols(symbols_path)

    default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
    error, changed_anything = filter_duplicates(default_params, symbols)

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


def get_regex_match_selection_parser(parser: ArgumentParser):
  parser.description = f"Select entries matching regex pattern."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("from_subsets", type=parse_non_empty_or_whitespace, nargs="+",
                      metavar="from-subsets", help="from subset", action=ConvertToOrderedSetAction)
  parser.add_argument("to_subset", type=parse_non_empty_or_whitespace, metavar="to-subset",
                      help="to subset")
  parser.add_argument("pattern", type=parse_non_empty_or_whitespace, metavar="REGEX",
                      help="to subset")
  return regex_match_selection


def regex_match_selection(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    symbols_path = get_data_symbols_path(data_folder)
    if not symbols_path.exists():
      logger.error(
        f"Symbols were not found! Skipping...")
      continue

    dataset = load_dataset(dataset_path)
    symbols = load_data_symbols(symbols_path)

    default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
    error, changed_anything = filter_regex_pattern(default_params, symbols, ns.pattern)

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
