import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Generator, List, Set, cast

from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection_cli.argparse_helper import parse_non_negative_integer
from txt_selection_cli.globals import SEL_ENC, SEL_EXT, SEL_LSEP, ExecutionResult
from txt_selection_cli.helper import (ConvertToOrderedSetAction, add_encoding_argument,
                                      get_all_files_in_all_subfolders, parse_existing_directory,
                                      parse_non_empty, parse_non_empty_or_whitespace, parse_path,
                                      try_read_selection, try_save_selection)
from txt_selection_cli.logging_configuration import get_file_logger, init_and_get_console_logger


def get_line_nr_moving_parser(parser: ArgumentParser):
  parser.description = "This command."
  parser.add_argument("directory", type=parse_existing_directory,
                      metavar="directory", help="directory containing files")
  parser.add_argument("from_set", type=parse_non_empty_or_whitespace,
                      metavar="from-set", help="from set")
  parser.add_argument("to_set", type=parse_non_empty_or_whitespace, metavar="to-set",
                      help="to set")
  parser.add_argument("line_numbers", type=parse_non_negative_integer, nargs="+", metavar="line-numbers",
                      help="ids to select", action=ConvertToOrderedSetAction)
  return select_line_nrs_ns


def select_line_nrs_ns(ns: Namespace) -> ExecutionResult:
  logger = init_and_get_console_logger(__name__)
  flogger = get_file_logger()

  logger.info("Searching for selection files...")
  sel_files: List[Path] = list(
    file
    for file in get_all_files_in_all_subfolders(ns.directory)
    if file.suffix.lower() == SEL_EXT.lower()
  )

  if len(sel_files) == 0:
    logger.info("Did not find any selection files!")
    return True, None

  projects: Set[Path] = {sel_file.parent for sel_file in sel_files}

  from_sel_files = {
    proj_dir: proj_dir / f"{ns.from_set}{SEL_EXT}"
    for proj_dir in projects
  }

  all_successfull = True
  changed_anything = False
  proj_path: Path

  for proj_path in tqdm(projects, desc="Processing projects(s)", unit=" project(s)"):
    flogger.info(f"Processing \"{proj_path}\"")

    from_selection_path = from_sel_files[proj_path]
    selection = try_read_selection(from_selection_path, flogger)
    if selection is None:
      flogger.info("Skipped.")
      all_successfull = False
      continue

    from_selection = selection

    to_selection_path = proj_path / f"{ns.to_set}{SEL_EXT}"
    if to_selection_path.exists():
      to_selection = try_read_selection(to_selection_path, flogger)
      if to_selection is None:
        flogger.info("Skipped.")
        all_successfull = False
        continue
    else:
      to_selection = OrderedSet()

    select_line_nrs = cast(OrderedSet, ns.line_numbers)

    missing_nrs = select_line_nrs.difference(from_selection)
    missing_nrs = missing_nrs.difference(to_selection)

    if len(missing_nrs) > 0:
      flogger.error(f"These line numbers do not exist in the from-set: {str(missing_nrs)}")
      flogger.info("Skipped.")
      all_successfull = False
      continue

    already_selected = select_line_nrs.intersection(to_selection)
    if len(already_selected) > 0:
      flogger.info(f"These line numbers are already in the to-set: {str(already_selected)}")

    existing_nrs = from_selection.intersection(select_line_nrs)
    if len(existing_nrs) == 0:
      flogger.info("Nothing to do.")
      continue

    flogger.info("Remove selection from from-set...")
    from_selection.difference_update(existing_nrs)
    flogger.info("Add selection to to-set...")
    to_selection.update(existing_nrs)

    success = try_save_selection(from_selection, from_selection_path, flogger)
    if not success:
      all_successfull = False
      continue

    changed_anything = True

    success = try_save_selection(to_selection, to_selection_path, flogger)
    if not success:
      all_successfull = False
      continue

  return all_successfull, changed_anything
