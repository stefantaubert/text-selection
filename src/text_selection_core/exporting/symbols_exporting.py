from typing import List, Optional, Tuple

from text_selection_core.types import Dataset, Lines, SubsetName
from text_selection_core.validation import (SubsetNotExistsError, SymbolsDoNotContainAllKeysError,
                                            ValidationError)


def export_symbols(dataset: Dataset, subset_name: SubsetName, data_symbols: Lines) -> Tuple[Optional[ValidationError], str]:
  if error := SubsetNotExistsError.validate(dataset, subset_name):
    return error, None

  if error := SymbolsDoNotContainAllKeysError.validate(dataset, data_symbols):
    return error, None

  subset = dataset.subsets[subset_name]
  strings = (
    data_symbols[data_id]
    for data_id in subset
  )

  result = "\n".join(strings)
  return None, result
