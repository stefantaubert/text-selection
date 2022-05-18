from logging import Logger
from typing import Optional, Tuple

from ordered_set import OrderedSet
from tqdm import tqdm
from text_selection_core.globals import TQDM_LINE_UNIT

from text_selection_core.types import Dataset, Lines, SubsetName
from text_selection_core.validation import (LinesCountNotMatchingError, SubsetNotExistsError,
                                            ValidationError)


def export_subset(dataset: Dataset, subset_names: OrderedSet[SubsetName], lines: Lines, lsep: str, logger: Logger) -> Tuple[Optional[ValidationError], str]:
  if error := SubsetNotExistsError.validate_names(dataset, subset_names):
    return error, None

  if error := LinesCountNotMatchingError.validate(dataset, lines):
    return error, None

  selected_lines = (
    lines[line_nr]
    for subset in subset_names
    for line_nr in dataset[subset]
  )

  selected_lines = tqdm(selected_lines, desc="Rejoining lines", unit=TQDM_LINE_UNIT)
  result = lsep.join(selected_lines)
  return None, result
