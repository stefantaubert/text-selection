from argparse import ArgumentParser, Namespace
from logging import Logger

from text_selection_cli.argparse_helper import (parse_non_empty_or_whitespace, parse_path)
from text_selection_cli.default_args import (add_file_arguments, add_project_argument)
from text_selection_cli.globals import ExecutionResult
from text_selection_cli.io_handling import try_load_dataset, try_load_file
from text_selection_core.exporting.symbols_exporting import export_symbols


def get_export_txt_parser(parser: ArgumentParser):
  parser.description = f"This command."
  add_project_argument(parser)
  parser.add_argument("subset", type=parse_non_empty_or_whitespace, metavar="subset",
                      help="subset which should be exported")
  add_file_arguments(parser)
  parser.add_argument("path", type=parse_path, metavar="path",
                      help="path to the exported text-file")
  return export_txt_ns


def export_txt_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.project, logger)
  if dataset is None:
    return False, None

  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if lines is None:
    return False, None

  logger.debug("Exporting...")
  error, text = export_symbols(dataset, ns.subset, lines, ns.lsep)

  success = error is None

  if not success:
    logger.error(f"{error.default_message}")
    return False, None

  assert text is not None
  logger.info(f"Saving output to \"{ns.path.absolute()}\"...")
  try:
    ns.path.write_text(text, ns.encoding)
  except Exception as ex:
    logger.error("Output couldn't be saved!")
    logger.exception(ex)
    return False, None

  return True, None
