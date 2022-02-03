from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import cast

from text_selection_core.weights.calculation import (
    divide_weights_inplace, get_character_count_weights, get_uniform_weights,
    get_word_count_weights)

from text_selection_app.argparse_helper import (get_optional,
                                                parse_existing_directory,
                                                parse_non_empty_or_whitespace,
                                                parse_non_negative_float,
                                                parse_positive_float)
from text_selection_app.helper import get_datasets
from text_selection_app.io_handling import (get_data_symbols_path,
                                            get_data_weights_path, load_data_symbols,
                                            load_data_weights, load_dataset,
                                            save_data_weights)


def get_uniform_weights_creation_parser(parser: ArgumentParser):
  parser.description = f"This command adds subsets."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("--name", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="name of the weights", default="weights")
  parser.add_argument("-o", "--overwrite", action="store_true",
                      help="overwrite weights")
  return create_uniform_weights_ns


def create_uniform_weights_ns(ns: Namespace) -> None:
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    weights_path = get_data_weights_path(data_folder, ns.name)

    if weights_path.is_file() and not ns.overwrite:
      logger.error("Weights already exist! Skipped.")
      continue

    dataset = load_dataset(dataset_path)

    weights = get_uniform_weights(dataset.ids)

    save_data_weights(weights_path, weights)


def get_word_count_weights_creation_parser(parser: ArgumentParser):
  parser.description = f"This command creates weights containing the word counts."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("--name", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="name of the weights", default="weights")
  parser.add_argument("-o", "--overwrite", action="store_true",
                      help="overwrite weights")
  return create_word_count_weights_ns


def create_word_count_weights_ns(ns: Namespace) -> None:
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    weights_path = get_data_weights_path(data_folder, ns.name)

    if weights_path.is_file() and not ns.overwrite:
      logger.error("Weights already exist! Skipped.")
      continue

    symbols_path = get_data_symbols_path(data_folder)
    if not symbols_path.exists():
      logger.error(
        f"Symbols were not found! Skipping...")
      continue

    symbols = load_data_symbols(symbols_path)

    weights = get_word_count_weights(symbols)

    save_data_weights(weights_path, weights)


def get_symbol_count_weights_creation_parser(parser: ArgumentParser):
  parser.description = f"This command creates weights containing the symbol counts."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("--name", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="name of the weights", default="weights")
  parser.add_argument("-o", "--overwrite", action="store_true",
                      help="overwrite weights")
  return create_symbol_count_weights_ns


def create_symbol_count_weights_ns(ns: Namespace) -> None:
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    weights_path = get_data_weights_path(data_folder, ns.name)

    if weights_path.is_file() and not ns.overwrite:
      logger.error("Weights already exist! Skipped.")
      continue

    symbols_path = get_data_symbols_path(data_folder)
    if not symbols_path.exists():
      logger.error(
        f"Symbols were not found! Skipping...")
      continue

    symbols = load_data_symbols(symbols_path)

    weights = get_character_count_weights(symbols)

    save_data_weights(weights_path, weights)


def get_weights_division_parser(parser: ArgumentParser):
  parser.description = f"This command creates weights containing the symbol counts."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("divisor", type=parse_positive_float, metavar="divisor",
                      help="divisor")
  parser.add_argument("--name", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="name of the weights", default="weights")
  parser.add_argument("--new-name", type=get_optional(parse_non_empty_or_whitespace), metavar="NAME",
                      help="custom new name of the weights", default=None)
  parser.add_argument("-o", "--overwrite", action="store_true",
                      help="overwrite weights")
  return create_weights_division_ns


def create_weights_division_ns(ns: Namespace) -> None:
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    weights_path = get_data_weights_path(data_folder, ns.name)

    if not weights_path.is_file():
      logger.error("Weights were not found! Skipped.")
      continue

    new_name: str = ns.name if ns.new_name is None else ns.new_name
    target_weights_path = get_data_weights_path(data_folder, new_name)

    if target_weights_path.is_file() and not ns.overwrite:
      logger.error("Weights already exist! Skipped.")
      continue

    weights = load_data_weights(weights_path)

    divide_weights_inplace(weights, ns.divisor)

    save_data_weights(target_weights_path, weights)
