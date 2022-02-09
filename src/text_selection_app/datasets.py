from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from shutil import copy2, rmtree
from typing import cast
from text_selection_app.default_args import add_encoding_argument, add_string_format_argument

from text_selection_core.datasets import create_from_text

from text_selection_app.argparse_helper import (parse_codec,
                                                parse_existing_file,
                                                parse_non_empty_or_whitespace,
                                                parse_path)
from text_selection_app.helper import get_datasets
from text_selection_app.io_handling import (get_data_symbols_path, get_dataset_path,
                                            save_data_symbols, save_dataset)


def get_dataset_creation_from_text_parser(parser: ArgumentParser):
  parser.description = f"This command reads the lines of a textfile and creates a dataset from it."
  parser.add_argument("directory", type=parse_path, metavar="directory",
                      help="directory to write")
  parser.add_argument("text", type=parse_existing_file, metavar="text",
                      help="path to textfile")
  add_encoding_argument(parser, "encoding of text")
  add_string_format_argument(parser, "text")
  parser.add_argument("--name", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="name of the initial subset containing all Id's", default="base")
  parser.add_argument("-o", "--overwrite", action="store_true",
                      help="overwrite complete directory")
  return create_dataset_from_text_ns


def create_dataset_from_text_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  data_folder = cast(Path, ns.directory)

  if data_folder.is_dir() and not ns.overwrite:
    logger.error("Directory already exists!")
    return

  lines = cast(Path, ns.text).read_text(ns.encoding).splitlines()

  error, result = create_from_text(lines, ns.name, ns.formatting)

  success = error is None

  if not success:
    logger.error(f"{error.default_message}")
  else:
    dataset, data_symbols = result

    if data_folder.is_dir():
      rmtree(data_folder)

    save_dataset(get_dataset_path(data_folder), dataset)
    save_data_symbols(get_data_symbols_path(data_folder), data_symbols)


def get_backup_parser(parser: ArgumentParser):
  parser.description = f"This command creates a backup of the database."
  parser.add_argument("directory", type=parse_path, metavar="directory",
                      help="directory to write")
  parser.add_argument("--name", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="name of the backup", default="backup")
  parser.add_argument("-o", "--overwrite", action="store_true",
                      help="overwrite backup if it exists")
  return backup_ns


def backup_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    backup_path = data_folder / f"{ns.name}.pkl"
    if backup_path.is_file() and not ns.overwrite:
      logger.error("Backup already exist! Skipped.")
      continue

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
  datasets = get_datasets(root_folder)

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
