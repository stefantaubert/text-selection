from logging import Logger

from text_selection_core.globals import ExecutionResult
from text_selection_core.types import Dataset, LineNrs, SubsetName, move_lines_to_subset


def filter_line_nrs(dataset: Dataset, to_subset_name: SubsetName, nrs: LineNrs, logger: Logger) -> ExecutionResult:
  if to_subset_name in dataset.subsets:
    to_subset = dataset.subsets[to_subset_name]
    already_selected = nrs.intersection(to_subset)
    if len(already_selected) > 0:
      logger.info(f"Lines {', '.join(already_selected)} were already selected!")

  changed_anything = False
  if len(nrs) > 0:
    logger.debug(f"Filtered {len(nrs)} numbers.")
    move_lines_to_subset(dataset, nrs, to_subset_name, logger)
    changed_anything = True
  return changed_anything
