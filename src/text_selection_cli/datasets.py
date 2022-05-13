from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from shutil import copy2
from typing import cast

from text_selection_cli.argparse_helper import (parse_existing_file, parse_non_empty,
                                                parse_non_empty_or_whitespace, parse_path)
from text_selection_cli.default_args import add_encoding_argument, add_file_arguments
from text_selection_cli.helper import get_datasets
from text_selection_cli.io_handling import get_dataset_path, try_load_file, try_save_dataset
from text_selection_core.types import create_dataset_from_line_count


def get_init_parser(parser: ArgumentParser):
  parser.description = "This command reads the lines of a textfile and initializes a dataset from it."
  parser.add_argument("directory", type=parse_path, metavar="directory",
                      help="directory to write")
  add_file_arguments(parser)
  parser.add_argument("name", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="name of the initial subset containing all line numbers")
  return create_dataset_from_text_ns


def create_dataset_from_text_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  data_folder = cast(Path, ns.directory)

  lines = try_load_file(data_folder / ns.file, ns.encoding, ns.lsep, logger)
  if lines is None:
    logger.info("Skipped!")
    return False, None

  logger.info("Creating dataset...")
  dataset = create_dataset_from_line_count(len(lines), ns.name)

  try_save_dataset(get_dataset_path(data_folder), dataset, logger)


def get_backup_parser(parser: ArgumentParser):
  parser.description = "This command creates a backup of the database."
  parser.add_argument("directory", type=parse_path, metavar="directory",
                      help="directory to write")
  parser.add_argument("--name", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="name of the backup", default="backup")
  return backup_ns


def backup_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder, logger)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    backup_path = data_folder / f"{ns.name}.pkl"

    copy2(dataset_path, backup_path)


def get_restore_parser(parser: ArgumentParser):
  parser.description = f"This command creates a backup of the database."
  parser.add_argument("directory", type=parse_path, metavar="directory",
                      help="directory to write")
  parser.add_argument("--name", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="name of the backup", default="backup")
  return restore_ns


def restore_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder, logger)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    backup_path = data_folder / f"{ns.name}.pkl"
    if not backup_path.is_file():
      logger.error("Backup does not exist! Skipped.")
      continue

    copy2(backup_path, dataset_path)
