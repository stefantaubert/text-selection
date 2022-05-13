from argparse import ArgumentParser, Namespace
from logging import Logger, getLogger
from pathlib import Path
from shutil import copy2
from typing import cast

from text_selection_cli.argparse_helper import (parse_existing_file, parse_non_empty,
                                                parse_non_empty_or_whitespace, parse_path)
from text_selection_cli.default_args import (add_encoding_argument, add_file_arguments,
                                             add_project_argument)
from text_selection_cli.globals import ExecutionResult
from text_selection_cli.helper import get_datasets
from text_selection_cli.io_handling import (get_dataset_path, try_load_dataset, try_load_file,
                                            try_save_dataset)
from text_selection_cli.logging_configuration import get_file_logger, init_and_return_loggers
from text_selection_core.types import create_dataset_from_line_count


def get_init_parser(parser: ArgumentParser):
  parser.description = "This command reads the lines of a textfile and initializes a dataset from it."
  add_project_argument(parser)
  add_file_arguments(parser)
  parser.add_argument("name", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="name of the initial subset containing all line numbers")
  return create_dataset_from_text_ns


def create_dataset_from_text_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  data_folder = cast(Path, ns.directory)

  lines = try_load_file(data_folder / ns.file, ns.encoding, ns.lsep, logger)
  if lines is None:
    logger.info("Skipped!")
    return False, None

  logger.info("Creating dataset...")
  dataset = create_dataset_from_line_count(len(lines), ns.name)

  try_save_dataset(get_dataset_path(data_folder), dataset, logger)

  return True, None
