import math
from argparse import ArgumentParser, Namespace
from logging import Logger, getLogger
from pathlib import Path
from typing import cast

from text_selection.selection import SelectionMode
from text_selection_cli.argparse_helper import (ConvertToOrderedSetAction, parse_existing_directory,
                                                parse_existing_file, parse_non_empty,
                                                parse_non_empty_or_whitespace,
                                                parse_non_negative_float,
                                                parse_non_negative_integer)
from text_selection_cli.default_args import (add_file_arguments, add_from_and_to_subsets_arguments,
                                             add_project_argument, add_to_subset_argument,
                                             parse_weights_name)
from text_selection_cli.globals import ExecutionResult
from text_selection_cli.helper import get_datasets
from text_selection_cli.io_handling import (get_data_weights_path, get_dataset_path,
                                            try_load_data_weights, try_load_dataset, try_load_file,
                                            try_save_dataset)
from text_selection_cli.logging_configuration import get_file_logger, init_and_return_loggers
from text_selection_core.common import SelectionDefaultParameters, WeightSelectionParameters
from text_selection_core.selection.fifo_selection import original_mode, select_fifo, subset_mode
from text_selection_core.selection.greedy_selection import GreedySelectionParameters, select_greedy
from text_selection_core.selection.kld_selection import KldSelectionParameters, select_kld
from text_selection_core.selection.nr_selection import select_ids


def get_id_selection_parser(parser: ArgumentParser):
  parser.description = "Select lines."
  add_project_argument(parser)
  add_to_subset_argument(parser)
  parser.add_argument("ids", type=parse_non_negative_integer, nargs="+", metavar="ids",
                      help="ids to select", action=ConvertToOrderedSetAction)
  return select_ids_from_ns


def select_ids_from_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.project, logger)
  if dataset is None:
    return False, False

  error, changed_anything = select_ids(dataset, ns.to_subset, ns.ids)

  success = error is None

  if not success:
    logger.error(f"{error.default_message}")
    return False, False

  if changed_anything:
    success = try_save_dataset(ns.project, dataset, logger)
    if not success:
      return False, False

  return True, changed_anything


def get_fifo_selection_parser(parser: ArgumentParser):
  parser.description = "Select lines by FIFO principle."
  add_project_argument(parser)
  add_from_and_to_subsets_arguments(parser)
  parser.add_argument("--mode", type=parse_non_empty_or_whitespace, metavar="MODE",
                      help="mode", default="subset", choices=[subset_mode, original_mode])
  add_termination_criteria_arguments(parser)
  return select_fifo_from_ns


def select_fifo_from_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.project, logger)
  if dataset is None:
    return False, False

  weights = try_load_data_weights(ns.weights, logger)
  if weights is None:
    return False, False

  default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
  weights_params = WeightSelectionParameters(
    weights, ns.limit, ns.limit_include_already_selected, ns.limit_percent)
  error, changed_anything = select_fifo(default_params, weights_params, ns.mode)

  success = error is None

  if not success:
    logger.error(error.default_message)
    return False, False

  if changed_anything:
    success = try_save_dataset(ns.project, dataset, logger)
    if not success:
      return False, False

  return True, changed_anything


def add_termination_criteria_arguments(parser: ArgumentParser) -> None:
  group = parser.add_argument_group("termination criteria arguments")
  group.add_argument("weights", type=parse_existing_file, metavar="weights",
                     help="weights name")
  group.add_argument("--limit", type=parse_non_negative_float, metavar="FLOAT",
                     help="weights limit", default=math.inf)
  group.add_argument("-i", "--limit-include-already-selected", action="store_true",
                     help="include already selected lines for limit")
  group.add_argument("-p", "--limit-percent", action="store_true",
                     help="limit is percentual; in this case it needs to be in interval (0, 100]")


def get_greedy_selection_parser(parser: ArgumentParser):
  parser.description = "Select lines by greedy principle."
  add_project_argument(parser)
  add_from_and_to_subsets_arguments(parser)
  add_file_arguments(parser, True)
  parser.add_argument("--include-selected", action="store_true",
                      help="consider already selected for the selection")
  add_termination_criteria_arguments(parser)
  return greedy_selection_ns


def greedy_selection_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.project, logger)
  if dataset is None:
    return False, False

  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if lines is None:
    return False, False

  weights = try_load_data_weights(ns.weights, logger)
  if weights is None:
    return False, False

  default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
  params = GreedySelectionParameters(lines, ns.sep, ns.include_selected, SelectionMode.FIRST)
  weights_params = WeightSelectionParameters(
    weights, ns.limit, ns.limit_include_already_selected, ns.limit_percent)

  error, changed_anything = select_greedy(default_params, params, weights_params)

  success = error is None

  if not success:
    logger.error(f"{error.default_message}")
    return False, False

  if changed_anything:
    success = try_save_dataset(ns.project, dataset, logger)
    if not success:
      return False, False

  return True, changed_anything


def get_kld_selection_parser(parser: ArgumentParser):
  parser.description = "Select lines by KLD principle."
  add_project_argument(parser)
  add_from_and_to_subsets_arguments(parser)
  add_file_arguments(parser, True)
  parser.add_argument("--include-selected", action="store_true",
                      help="consider already selected for the selection")
  add_termination_criteria_arguments(parser)
  return kld_selection_ns


def kld_selection_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.project, logger)
  if dataset is None:
    return False, False

  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if lines is None:
    return False, False

  weights = try_load_data_weights(ns.weights, logger)
  if weights is None:
    return False, False

  default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
  params = KldSelectionParameters(lines, ns.sep, ns.include_selected, SelectionMode.FIRST)
  weights_params = WeightSelectionParameters(
    weights, ns.limit, ns.limit_include_already_selected, ns.limit_percent)

  error, changed_anything = select_kld(default_params, params, weights_params)

  success = error is None

  if not success:
    logger.error(f"{error.default_message}")
    return False, False

  if changed_anything:
    success = try_save_dataset(ns.project, dataset, logger)
    if not success:
      return False, False

  return True, changed_anything
