import math
from argparse import ArgumentParser, Namespace
from logging import Logger

from text_selection_cli.argparse_helper import (ConvertToSetAction, get_optional,
                                                parse_existing_file, parse_integer_greater_one,
                                                parse_non_empty, parse_non_negative_float,
                                                parse_positive_float, parse_positive_integer)
from text_selection_cli.default_args import (add_dataset_argument, add_dry_argument,
                                             add_file_arguments, add_from_and_to_subsets_arguments)
from text_selection_cli.globals import ExecutionResult
from text_selection_cli.io_handling import (try_load_data_weights, try_load_dataset, try_load_file,
                                            try_save_dataset)
from text_selection_core.common import SelectionDefaultParameters
from text_selection_core.filtering.duplicates_filter import filter_duplicates
from text_selection_core.filtering.line_unit_frequency_filter import (
  LineUnitFrequencyFilterParameters, filter_lines_with_line_unit_frequencies)
from text_selection_core.filtering.regex_filter import filter_regex_pattern
from text_selection_core.filtering.string_filter import filter_by_string
from text_selection_core.filtering.unit_frequency_filter import (CountFilterParameters,
                                                                 filter_lines_with_unit_frequencies)
from text_selection_core.filtering.vocabulary_filtering import (
  VocabularyFilterParameters, filter_lines_with_vocabulary_frequencies)
from text_selection_core.filtering.weights_filter import WeightsFilterParameters, filter_weights
from text_selection_core.validation import ValidationErrBase


def get_duplicate_selection_parser(parser: ArgumentParser):
  parser.description = "Select duplicate entries."
  add_dataset_argument(parser)
  add_from_and_to_subsets_arguments(parser)
  add_file_arguments(parser)
  add_dry_argument(parser)
  return select_duplicates_ns


def select_duplicates_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.dataset, logger)
  if isinstance(dataset, ValidationErrBase):
    return dataset

  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if isinstance(lines, ValidationErrBase):
    return lines

  default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
  changed_anything = filter_duplicates(default_params, lines, flogger)
  if isinstance(changed_anything, ValidationErrBase):
    return changed_anything

  if changed_anything and not ns.dry:
    if error := try_save_dataset(ns.dataset, dataset, logger):
      return error

  return changed_anything


def get_regex_match_selection_parser(parser: ArgumentParser):
  parser.description = "Select entries matching regex pattern."
  add_dataset_argument(parser)
  add_from_and_to_subsets_arguments(parser)
  add_file_arguments(parser)
  parser.add_argument("pattern", type=parse_non_empty, metavar="REGEX",
                      help="to subset")
  add_dry_argument(parser)
  return regex_match_selection


def regex_match_selection(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.dataset, logger)
  if isinstance(dataset, ValidationErrBase):
    return dataset

  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if isinstance(lines, ValidationErrBase):
    return lines

  default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
  changed_anything = filter_regex_pattern(default_params, lines, ns.pattern, flogger)
  if isinstance(changed_anything, ValidationErrBase):
    return changed_anything

  if changed_anything and not ns.dry:
    if error := try_save_dataset(ns.dataset, dataset, logger):
      return error

  return changed_anything


def get_string_filter_parser(parser: ArgumentParser):
  parser.description = "Select entries matching regex pattern."
  add_dataset_argument(parser)
  add_from_and_to_subsets_arguments(parser)
  add_file_arguments(parser)
  parser.add_argument("--starts-with", type=parse_non_empty, metavar="STRING",
                      nargs="*", help="lines that starts with these strings", default={}, action=ConvertToSetAction)
  parser.add_argument("--ends-with", type=parse_non_empty, metavar="STRING",
                      nargs="*", help="lines that ends with these strings", default={}, action=ConvertToSetAction)
  parser.add_argument("--contains", type=parse_non_empty, metavar="STRING",
                      nargs="*", help="lines that contains these strings", default={}, action=ConvertToSetAction)
  parser.add_argument("--equals", type=str, metavar="STRING", nargs="*",
                      help="lines that are equal with these strings", default={}, action=ConvertToSetAction)
  parser.add_argument("--mode", type=str, choices=[
                      "all", "any"], help="mode to evaluate: all => all arguments need to match; any => any arguments needs to match", default="any")
  add_dry_argument(parser)
  return filter_by_string_ns


def filter_by_string_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.dataset, logger)
  if isinstance(dataset, ValidationErrBase):
    return dataset

  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if isinstance(lines, ValidationErrBase):
    return lines

  default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
  changed_anything = filter_by_string(
    default_params, lines, ns.starts_with, ns.ends_with, ns.contains, ns.equals, ns.mode, flogger)
  if isinstance(changed_anything, ValidationErrBase):
    return changed_anything

  if changed_anything and not ns.dry:
    if error := try_save_dataset(ns.dataset, dataset, logger):
      return error

  return changed_anything


def get_unit_frequency_parser(parser: ArgumentParser):
  parser.description = "Select entries ..."
  add_dataset_argument(parser)
  add_from_and_to_subsets_arguments(parser)
  add_file_arguments(parser, True)
  parser.add_argument("min_count", type=parse_positive_integer, metavar="MIN-COUNT",
                      help="inclusive minimum count how often all units of a sentence should occur")
  parser.add_argument("--max-count", type=get_optional(parse_integer_greater_one), metavar="MAX-COUNT",
                      help="exclusive maximum count how often all units of a sentence should occur", nargs="?", const=None, default=None)
  parser.add_argument("--mode", type=str, choices=[
                      "all", "any"], help="mode to evaluate count boundaries: all => all units need to match; any => any unit needs to match", default="all")
  parser.add_argument("--all", action="store_true",
                      help="calculate occurrences in the total dataset; otherwise only the occurrences from the FROM-SUBSETs are counted")
  add_dry_argument(parser)
  return filter_unit_counts_ns


def filter_unit_counts_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.dataset, logger)
  if isinstance(dataset, ValidationErrBase):
    return dataset

  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if isinstance(lines, ValidationErrBase):
    return lines

  default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
  params = CountFilterParameters(lines, ns.sep, ns.min_count, ns.max_count, ns.all, ns.mode)
  changed_anything = filter_lines_with_unit_frequencies(default_params, params, flogger)
  if isinstance(changed_anything, ValidationErrBase):
    return changed_anything

  if changed_anything and not ns.dry:
    if error := try_save_dataset(ns.dataset, dataset, logger):
      return error

  return changed_anything


def get_line_unit_frequency_parser(parser: ArgumentParser):
  parser.description = "Select entries ..."
  add_dataset_argument(parser)
  add_from_and_to_subsets_arguments(parser)
  add_file_arguments(parser, True)
  parser.add_argument("--min-count", type=parse_positive_integer, metavar="MIN-COUNT",
                      help="inclusive minimum count how often all units of a sentence should occur", default=1)
  parser.add_argument("--max-count", type=get_optional(parse_integer_greater_one), metavar="MAX-COUNT",
                      help="exclusive maximum count how often all units of a sentence should occur", default=math.inf)
  parser.add_argument("--mode", type=str, choices=[
                      "all", "any"], help="mode to evaluate count boundaries: all => all units need to match; any => any unit needs to match", default="all")
  add_dry_argument(parser)
  return filter_line_unit_counts_ns


def filter_line_unit_counts_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.dataset, logger)
  if isinstance(dataset, ValidationErrBase):
    return dataset

  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if isinstance(lines, ValidationErrBase):
    return lines

  default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
  params = LineUnitFrequencyFilterParameters(lines, ns.sep, ns.min_count, ns.max_count, ns.mode)
  changed_anything = filter_lines_with_line_unit_frequencies(default_params, params, flogger)
  if isinstance(changed_anything, ValidationErrBase):
    return changed_anything

  if changed_anything and not ns.dry:
    if error := try_save_dataset(ns.dataset, dataset, logger):
      return error

  return changed_anything


def get_weight_filtering_parser(parser: ArgumentParser):
  parser.description = "Select entries ..."
  add_dataset_argument(parser)
  add_from_and_to_subsets_arguments(parser)
  parser.add_argument("weights", type=parse_existing_file, metavar="WEIGHTS-PATH",
                      help="path to the weights")
  parser.add_argument("--min-weight", type=parse_non_negative_float, metavar="MIN-WEIGHT",
                      help="inclusive minimum weight", default=0)
  parser.add_argument("--max-weight", type=parse_positive_float, metavar="MAX-WEIGHT",
                      help="exclusive maximum weight", default=math.inf)
  add_dry_argument(parser)
  return filter_weights_ns


def filter_weights_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.dataset, logger)
  if isinstance(dataset, ValidationErrBase):
    return dataset

  weights = try_load_data_weights(ns.weights, logger)
  if isinstance(weights, ValidationErrBase):
    return weights

  default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
  params = WeightsFilterParameters(weights, ns.min_weight, ns.max_weight)
  changed_anything = filter_weights(default_params, params, flogger)
  if isinstance(changed_anything, ValidationErrBase):
    return changed_anything

  if changed_anything and not ns.dry:
    if error := try_save_dataset(ns.dataset, dataset, logger):
      return error

  return changed_anything


def get_vocabulary_filtering_parser(parser: ArgumentParser):
  parser.description = "Select entries ..."
  add_dataset_argument(parser)
  add_from_and_to_subsets_arguments(parser)
  add_file_arguments(parser, True)
  parser.add_argument("vocabulary", type=parse_existing_file, metavar="VOCABULARY-PATH",
                      help="path to the vocabulary")
  parser.add_argument("--min-count", type=parse_non_negative_float, metavar="MIN-COUNT",
                      help="inclusive minimum count", default=0)
  parser.add_argument("--max-count", type=parse_positive_integer, metavar="MAX-COUNT",
                      help="exclusive maximum count", default=math.inf)
  parser.add_argument("--mode", type=str, choices=[
      "oov", "iv"], help="mode: oov => count of out-of-vocabulary units; iv => count of in-vocabulary units", default="oov")
  add_dry_argument(parser)
  return filter_by_vocabulary_ns


def filter_by_vocabulary_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.dataset, logger)
  if isinstance(dataset, ValidationErrBase):
    return dataset

  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if isinstance(lines, ValidationErrBase):
    return lines

  logger.info(f"Reading \"{ns.vocabulary.absolute()}\"...")
  try:
    vocabulary_text = ns.vocabulary.read_text(ns.encoding)
  except Exception as ex:
    logger.error("Vocabulary file couldn't be read!")
    logger.exception(ex)
    return None

  logger.info("Parsing vocabulary...")
  vocabulary = set(vocabulary_text.split(ns.lsep))
  logger.debug(f"Parsed {len(vocabulary)} vocabulary units.")

  logger.info("Filtering...")
  default_params = SelectionDefaultParameters(dataset, ns.from_subsets, ns.to_subset)
  params = VocabularyFilterParameters(
    lines, ns.sep, ns.min_count, ns.max_count, vocabulary, ns.mode)
  changed_anything = filter_lines_with_vocabulary_frequencies(
    default_params, params, flogger)
  if isinstance(changed_anything, ValidationErrBase):
    return changed_anything

  if changed_anything and not ns.dry:
    if error := try_save_dataset(ns.dataset, dataset, logger):
      return error

  return changed_anything
