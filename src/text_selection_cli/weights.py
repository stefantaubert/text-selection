from argparse import ArgumentParser, Namespace
from logging import Logger, getLogger
from pathlib import Path
from typing import cast

from text_selection_cli.argparse_helper import (get_optional, parse_existing_directory,
                                                parse_non_empty, parse_non_empty_or_whitespace,
                                                parse_positive_float)
from text_selection_cli.default_args import (add_directory_argument, add_encoding_argument,
                                             add_file_arguments, parse_weights_name)
from text_selection_cli.helper import get_datasets
from text_selection_cli.io_handling import (get_data_weights_path, try_load_data_weights,
                                            try_load_dataset, try_load_file, try_save_data_weights)
from text_selection_cli.logging_configuration import get_file_logger, init_and_return_loggers
from text_selection_core.globals import ExecutionResult
from text_selection_core.weights.calculation import (divide_weights_inplace, get_uniform_weights,
                                                     get_word_count_weights)


def get_uniform_weights_creation_parser(parser: ArgumentParser):
  parser.description = "This command creates uniform weights."
  add_directory_argument(parser)
  parser.add_argument("name", type=parse_weights_name, metavar="NAME",
                      help="name of the weights")
  return create_uniform_weights_ns


def create_uniform_weights_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder, logger)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    weights_path = get_data_weights_path(data_folder, ns.name)

    dataset = try_load_dataset(dataset_path, logger)
    if dataset is None:
      logger.info("Skipped!")
      continue

    weights = get_uniform_weights(dataset.get_line_nrs())

    try_save_data_weights(weights_path, weights, logger)


def get_word_count_weights_creation_parser(parser: ArgumentParser):
  parser.description = "This command creates weights containing the word/symbol counts."
  add_directory_argument(parser)
  add_file_arguments(parser, True)
  parser.add_argument("name", type=parse_weights_name, metavar="NAME",
                      help="name of the weights")
  return create_word_count_weights_ns


def create_word_count_weights_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder, logger)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    weights_path = get_data_weights_path(data_folder, ns.name)

    lines = try_load_file(data_folder / ns.file, ns.encoding, ns.lsep, logger)
    if lines is None:
      logger.info("Skipped!")
      continue

    logger.info("Calculating weights...")
    weights = get_word_count_weights(lines, ns.sep)

    try_save_data_weights(weights_path, weights, logger)


def get_weights_division_parser(parser: ArgumentParser):
  parser.description = "This command creates weights containing the ..."
  add_directory_argument(parser)
  parser.add_argument("name", type=parse_weights_name, metavar="NAME",
                      help="name of the weights")
  parser.add_argument("divisor", type=parse_positive_float, metavar="divisor",
                      help="divisor")
  parser.add_argument("--new-name", type=get_optional(parse_non_empty_or_whitespace), metavar="NAME",
                      help="custom new name of the weights", default=None)
  return create_weights_division_ns


def create_weights_division_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder, logger)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    new_name: str = ns.name if ns.new_name is None else ns.new_name
    target_weights_path = get_data_weights_path(data_folder, new_name)

    weights_path = get_data_weights_path(data_folder, ns.name)
    weights = try_load_data_weights(weights_path, logger)
    if weights is None:
      logger.info("Skipped.")
      continue

    divide_weights_inplace(weights, ns.divisor)

    try_save_data_weights(target_weights_path, weights, logger)
