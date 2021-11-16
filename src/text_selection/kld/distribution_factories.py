import numpy as np


class DistributionFactory():
  def get(self, count: int) -> np.ndarray:
    raise NotImplementedError()


class UniformDistributionFactory(DistributionFactory):
  def get(self, count: int) -> np.ndarray:
    if count == 0:
      result = np.zeros(shape=(0), dtype=np.float64)
      return result
    result = np.full(shape=(count), dtype=np.float64, fill_value=1)
    result = np.divide(result, count)
    return result
