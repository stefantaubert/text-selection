from typing import Iterator

from text_selection_core.types import LineNr, DataWeights, Weight


class WeightsIterator(Iterator[LineNr]):
  def __init__(self, iterator: Iterator[LineNr], weights: DataWeights, target: Weight, initial_weight: Weight) -> None:
    assert initial_weight >= 0

    super().__init__()
    self.__iterator = iterator
    self.__weights = weights
    self.__target = target
    self.__current_total = initial_weight
    self.__enough_data_was_available = True
    self.__update_progress = initial_weight

  @property
  def tqdm_update(self) -> int:
    return self.__update_progress

  @property
  def was_enough_data_available(self) -> bool:
    return self.__enough_data_was_available

  def __iter__(self) -> Iterator[int]:
    return self

  def __next__(self) -> int:
    if self.__current_total >= self.__target:
      raise StopIteration()

    try:
      selected_id = next(self.__iterator)
    except StopIteration:
      self.__enough_data_was_available = False
      raise StopIteration()

    assert selected_id in self.__weights
    selected_until_value = self.__weights[selected_id]
    new_total = self.__current_total + selected_until_value
    if new_total <= self.__target:
      self.__current_total = new_total
      self.__update_progress = selected_until_value
      return selected_id
    else:
      raise StopIteration()
