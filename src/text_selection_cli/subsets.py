import math
from argparse import ArgumentParser, Namespace
from logging import Logger, getLogger
from pathlib import Path
from typing import cast

from text_selection_cli.argparse_helper import (ConvertToOrderedSetAction, parse_existing_directory,
                                                parse_non_empty_or_whitespace)
from text_selection_cli.default_args import add_project_argument
from text_selection_cli.globals import ExecutionResult
from text_selection_cli.helper import get_datasets
from text_selection_cli.io_handling import get_dataset_path, try_load_dataset, try_save_dataset
from text_selection_cli.logging_configuration import get_file_logger, init_and_return_loggers
from text_selection_core.subsets import add_subsets, remove_subsets


def get_subsets_creation_parser(parser: ArgumentParser):
  parser.description = "This command adds subsets."
  add_project_argument(parser)
  parser.add_argument("names", type=parse_non_empty_or_whitespace, nargs="+", metavar="names",
                      help="names of subsets that should be added", action=ConvertToOrderedSetAction)
  return add_subsets_ns


def add_subsets_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.project, logger)
  if dataset is None:
    return False, False

  logger.info("Adding subset(s)...")
  error, changed_anything = add_subsets(dataset, ns.names)

  success = error is None

  if not success:
    logger.error(f"{error.default_message}")
    return False, False

  if changed_anything:
    success = try_save_dataset(ns.project, dataset, logger)
    if not success:
      return False, False

  return True, changed_anything


def get_subsets_removal_parser(parser: ArgumentParser):
  parser.description = "This command removes subsets."
  add_project_argument(parser)
  parser.add_argument("names", type=parse_non_empty_or_whitespace, nargs="+", metavar="names",
                      help="names of subsets that should be removed (Note: at least one subset needs to be left after removal)", action=ConvertToOrderedSetAction)
  return add_subsets_ns


def remove_subsets_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.project, logger)
  if dataset is None:
    return False, False

  logger.info("Removing subset(s)...")
  error, changed_anything = remove_subsets(dataset, ns.names)

  success = error is None

  if not success:
    logger.error(f"{error.default_message}")
    return False, False

  if changed_anything:
    success = try_save_dataset(ns.project, dataset, logger)
    if not success:
      return False, False

  return True, changed_anything
