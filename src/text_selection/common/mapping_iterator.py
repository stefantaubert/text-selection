from typing import Dict, Iterator


class MappingIterator(Iterator[int]):
  def __init__(self, iterator: Iterator[int], mapping: Dict[int, int]) -> None:
    super().__init__()
    self.iterator = iterator
    self.mapping = mapping

  def __iter__(self) -> Iterator[int]:
    return self

  def __next__(self) -> int:
    result = next(self.iterator)
    assert result in self.mapping
    mapped_result = self.mapping[result]
    return mapped_result
