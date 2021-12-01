from typing import Dict, Iterator, List, Tuple, Union

from tqdm import tqdm


def iterate_durations_dict(iterator: Iterator[int], until_values: Dict[int, Union[int, float]], until_value: Union[int, float]) -> Tuple[List[int], bool]:
  iterated_values: List[int] = []
  enough_data_was_available = False
  max_until = sum(until_values)
  adjusted_until = round(min(until_value, max_until))
  current_total = 0.0
  with tqdm(total=len(until_values), initial=0) as progress_bar1:
    with tqdm(total=adjusted_until, initial=round(current_total)) as progress_bar:
      for selected_key in iterator:
        assert selected_key in until_values
        selected_until_value = until_values[selected_key]
        new_total = current_total + selected_until_value
        if new_total <= until_value:
          iterated_values.append(selected_key)
          current_total = new_total
          progress_bar.update(round(selected_until_value))
          progress_bar1.update()
          if current_total == until_value:
            enough_data_was_available = True
        else:
          enough_data_was_available = True
          break
    # Selected: {current_total:.1f}/{until_value:.1f} ({current_total/until_value*100:.2f}%).
    return iterated_values, enough_data_was_available


# def iterate_durations(iterator: Iterator[int], until_values: np.ndarray, until_value: Union[int, float]) -> Tuple[List[int], bool]:
#   iterated_values: List[int] = []
#   enough_data_was_available = False
#   max_until = sum(until_values)
#   adjusted_until = round(min(until_value, max_until))
#   current_total = 0.0
#   with tqdm(total=adjusted_until, initial=round(current_total)) as progress_bar:
#     for selected_key in iterator:
#       assert 0 <= selected_key < len(until_values)
#       selected_until_value = until_values[selected_key]
#       new_total = current_total + selected_until_value
#       if new_total <= until_value:
#         iterated_values.append(selected_key)
#         current_total = new_total
#         progress_bar.update(round(selected_until_value))
#         if current_total == until_value:
#           enough_data_was_available = True
#       else:
#         enough_data_was_available = True
#         break
#   # Selected: {current_total:.1f}/{until_value:.1f} ({current_total/until_value*100:.2f}%).
#   return iterated_values, enough_data_was_available
