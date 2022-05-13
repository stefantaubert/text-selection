from argparse import ArgumentParser, Namespace
from logging import Logger

from text_selection_cli.argparse_helper import parse_non_empty_or_whitespace
from text_selection_cli.default_args import (add_file_arguments, add_from_and_to_subsets_arguments,
                                             add_project_argument)
from text_selection_cli.globals import ExecutionResult
from text_selection_cli.io_handling import (try_load_dataset, try_load_file, try_save_dataset)
from text_selection_core.common import SelectionDefaultParameters
from text_selection_core.filtering.duplicates_filter import filter_duplicates
from text_selection_core.filtering.regex_filter import filter_regex_pattern


def get_duplicate_selection_parser(parser: ArgumentParser):
  parser.description = "Select duplicate entries."
  add_project_argument(parser)
  add_file_arguments(parser)
  add_from_and_to_subsets_arguments(parser)
  return select_duplicates_ns


def select_duplicates_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.project, logger)
  if dataset is None:
    return False, False

  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if lines is None:
    return False, False

  default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
  error, changed_anything = filter_duplicates(default_params, lines)

  success = error is None

  if not success:
    logger.error(f"{error.default_message}")
    return False, False

  if changed_anything:
    success = try_save_dataset(ns.project, dataset, logger)
    if not success:
      return False, False

  return True, changed_anything


def get_regex_match_selection_parser(parser: ArgumentParser):
  parser.description = "Select entries matching regex pattern."
  add_project_argument(parser)
  add_file_arguments(parser)
  add_from_and_to_subsets_arguments(parser)
  parser.add_argument("pattern", type=parse_non_empty_or_whitespace, metavar="REGEX",
                      help="to subset")
  return regex_match_selection


def regex_match_selection(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.project, logger)
  if dataset is None:
    return False, False

  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if lines is None:
    return False, False

  default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
  error, changed_anything = filter_regex_pattern(default_params, lines, ns.pattern)

  success = error is None

  if not success:
    logger.error(f"{error.default_message}")
    return False, False

  if changed_anything:
    success = try_save_dataset(ns.project, dataset, logger)
    if not success:
      return False, False

  return True, changed_anything
