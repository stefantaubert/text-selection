from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import cast

from text_selection_app.argparse_helper import (ConvertToOrderedSetAction, parse_existing_directory,
                                                parse_non_empty, parse_non_empty_or_whitespace)
from text_selection_app.helper import get_datasets
from text_selection_app.io_handling import get_dataset_path, load_dataset, save_dataset
from text_selection_core.common import SelectionDefaultParameters
from text_selection_core.filtering.duplicates_filter import filter_duplicates
from text_selection_core.filtering.regex_filter import filter_regex_pattern


def get_duplicate_selection_parser(parser: ArgumentParser):
  parser.description = "Select duplicate entries."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("file", type=parse_non_empty_or_whitespace,
                      help="name of the file containing the lines")
  parser.add_argument("--lsep", type=parse_non_empty, default="\n",
                      help="line separator")
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

    symbols_path = data_folder / cast(str, ns.name)

    if not symbols_path.exists():
      logger.error(
        "Symbols were not found! Skipping...")
      continue

    dataset = load_dataset(dataset_path)
    lines = symbols_path.read_text(ns.encoding).split(ns.lsep)

    default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
    error, changed_anything = filter_duplicates(default_params, lines)

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
  parser.description = "Select entries matching regex pattern."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("file", type=parse_non_empty_or_whitespace,
                      help="name of the file containing the lines")
  parser.add_argument("--lsep", type=parse_non_empty, default="\n",
                      help="line separator")
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

    symbols_path = data_folder / cast(str, ns.name)

    if not symbols_path.exists():
      logger.error(
        "Symbols were not found! Skipping...")
      continue

    dataset = load_dataset(dataset_path)
    lines = symbols_path.read_text(ns.encoding).split(ns.lsep)

    default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
    error, changed_anything = filter_regex_pattern(default_params, lines, ns.pattern)

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
