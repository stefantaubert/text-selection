from logging import Logger
from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import cast

from text_selection_cli.argparse_helper import (parse_existing_directory,
                                                parse_non_empty_or_whitespace)
from text_selection_cli.default_args import add_directory_argument
from text_selection_cli.helper import get_datasets
from text_selection_cli.io_handling import get_dataset_path, try_load_dataset, try_save_dataset
from text_selection_cli.logging_configuration import get_file_logger, init_and_return_loggers
from text_selection_core.subsets import rename_subset


def get_subset_renaming_parser(parser: ArgumentParser):
  parser.description = "This command rename a subset."
  add_directory_argument(parser)
  parser.add_argument("name", type=parse_non_empty_or_whitespace, metavar="name",
                      help="subset that should be renamed")
  parser.add_argument("new_name", type=parse_non_empty_or_whitespace, metavar="new-name",
                      help="new name")
  return rename_subsets_ns


def rename_subsets_ns(ns: Namespace, logger: Logger, flogger: Logger):
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

    logger.info("Renaming subset...")
    error, changed_anything = rename_subset(dataset, ns.name, ns.new_name)

    success = error is None

    if not success:
      logger.error(f"{error.default_message}")
      logger.info("Skipped.")
      assert not changed_anything
    else:
      assert changed_anything
      try_save_dataset(get_dataset_path(data_folder), dataset, logger)
