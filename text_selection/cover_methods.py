from collections import OrderedDict
from typing import Dict
from typing import OrderedDict as OrderedDictType
from typing import Set, TypeVar, Union

from ordered_set import OrderedSet
from tqdm import tqdm

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def cover_default(data: OrderedDictType[_T1, Set[_T2]], to_cover: Set[_T2]) -> OrderedSet[_T1]:
  assert isinstance(data, OrderedDict)
  result: OrderedSet[_T1] = OrderedSet([
    k for k, v in tqdm(data.items()) if len(v.intersection(to_cover)) > 0
  ])
  return result
