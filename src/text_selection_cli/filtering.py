from argparse import ArgumentParser, Namespace
from logging import Logger, getLogger
from pathlib import Path
from typing import cast

from text_selection_cli.argparse_helper import (ConvertToOrderedSetAction, parse_existing_directory,
                                                parse_non_empty, parse_non_empty_or_whitespace)
from text_selection_cli.default_args import (add_directory_argument, add_encoding_argument,
                                             add_file_arguments, add_from_and_to_subsets_arguments)
from text_selection_cli.helper import get_datasets
from text_selection_cli.io_handling import (get_dataset_path, try_load_dataset, try_load_file,
                                            try_save_dataset)
from text_selection_cli.logging_configuration import get_file_logger, init_and_return_loggers
from text_selection_core.common import SelectionDefaultParameters
from text_selection_core.filtering.duplicates_filter import filter_duplicates
from text_selection_core.filtering.regex_filter import filter_regex_pattern


def get_duplicate_selection_parser(parser: ArgumentParser):
  parser.description = "Select duplicate entries."
  add_directory_argument(parser)
  add_file_arguments(parser)
  add_from_and_to_subsets_arguments(parser)
  return select_duplicates_ns


def select_duplicates_ns(ns: Namespace, logger: Logger, flogger: Logger):
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder, logger)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    dataset = try_load_dataset(dataset_path, logger)
    if dataset is None:
      logger.info("Skipped!")
      continue

    lines = try_load_file(data_folder / ns.file, ns.encoding, ns.lsep, logger)
    if lines is None:
      logger.info("Skipped!")
      continue

    default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
    error, changed_anything = filter_duplicates(default_params, lines)

    success = error is None

    if not success:
      logger.error(f"{error.default_message}")
      logger.info("Skipped.")
      assert not changed_anything
    else:
      if changed_anything:
        try_save_dataset(get_dataset_path(data_folder), dataset, logger)
      else:
        logger.info("Didn't changed anything!")


def get_regex_match_selection_parser(parser: ArgumentParser):
  parser.description = "Select entries matching regex pattern."
  add_directory_argument(parser)
  add_file_arguments(parser)
  add_from_and_to_subsets_arguments(parser)
  parser.add_argument("pattern", type=parse_non_empty_or_whitespace, metavar="REGEX",
                      help="to subset")
  return regex_match_selection


def regex_match_selection(ns: Namespace, logger: Logger, flogger: Logger):
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder, logger)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    dataset = try_load_dataset(dataset_path, logger)
    if dataset is None:
      logger.info("Skipped!")
      continue

    lines = try_load_file(data_folder / ns.file, ns.encoding, ns.lsep, logger)
    if lines is None:
      logger.info("Skipped!")
      continue

    default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
    error, changed_anything = filter_regex_pattern(default_params, lines, ns.pattern)

    success = error is None

    if not success:
      logger.error(f"{error.default_message}")
      logger.info("Skipped.")
      assert not changed_anything
    else:
      if changed_anything:
        try_save_dataset(get_dataset_path(data_folder), dataset, logger)
      else:
        logger.info("Didn't changed anything!")
