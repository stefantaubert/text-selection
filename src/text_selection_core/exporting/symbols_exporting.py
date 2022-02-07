from typing import List, Optional, Tuple
from text_selection_core.types import DataSymbols, Dataset, SubsetName, item_to_symbols
from text_selection_core.validation import SubsetNotExistsError, SymbolsDoNotContainAllKeysError, ValidationError
from text_utils import StringFormat2


def export_symbols(dataset: Dataset, subset_name: SubsetName, data_symbols: DataSymbols, string_format: StringFormat2) -> Tuple[Optional[ValidationError], str]:
  if error := SubsetNotExistsError.validate(dataset, subset_name):
    return error, None

  if error := SymbolsDoNotContainAllKeysError.validate(dataset, data_symbols):
    return error, None

  subset = dataset.subsets[subset_name]
  strings = (
    string_format.convert_symbols_to_string(item_to_symbols(data_symbols[data_id]))
    for data_id in subset
  )

  result = "\n".join(strings)
  return None, result
