from text_selection_core.types import DataIds, DataWeights


def get_uniform_weights(ids: DataIds) -> DataWeights:
  result = dict.fromkeys(ids, 1)
  return result
