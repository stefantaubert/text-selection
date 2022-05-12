import math
from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import cast

from text_selection_core.subsets import add_subsets, remove_subsets

from text_selection_app.argparse_helper import (ConvertToOrderedSetAction,
                                                parse_existing_directory,
                                                parse_non_empty_or_whitespace)
from text_selection_app.helper import get_datasets
from text_selection_app.io_handling import get_dataset_path, load_dataset, save_dataset


def get_subsets_creation_parser(parser: ArgumentParser):
  parser.description = "This command adds subsets."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("names", type=parse_non_empty_or_whitespace, nargs="+", metavar="names",
                      help="names of subsets that should be added", action=ConvertToOrderedSetAction)
  return add_subsets_ns


def add_subsets_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    dataset = load_dataset(dataset_path)

    error, changed_anything = add_subsets(dataset, ns.names)

    success = error is None

    if not success:
      logger.error(f"{error.default_message}")
      logger.info("Skipped.")
      assert not changed_anything
    else:
      assert changed_anything
      save_dataset(get_dataset_path(data_folder), dataset)


def get_subsets_removal_parser(parser: ArgumentParser):
  parser.description = "This command removes subsets."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("names", type=parse_non_empty_or_whitespace, nargs="+", metavar="names",
                      help="names of subsets that should be removed (Note: at least one subset needs to be left after removal)", action=ConvertToOrderedSetAction)
  return add_subsets_ns


def remove_subsets_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    dataset = load_dataset(dataset_path)

    error, changed_anything = remove_subsets(dataset, ns.names)

    success = error is None

    if not success:
      logger.error(f"{error.default_message}")
      logger.info("Skipped.")
      assert not changed_anything
    else:
      assert changed_anything
      save_dataset(get_dataset_path(data_folder), dataset)
